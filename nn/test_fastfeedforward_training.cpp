TEST_F(FastFeedforwardTest, SimpleTrainingTest) {
    // Create a small network for testing
    gpt::FastFeedforwardNetwork network(4, 4);
    
    // Prepare input and target output data
    nn::Parameter x_train(nn::DT_FLOAT, 2 * 4);
    nn::Parameter y_true(nn::DT_FLOAT, 2 * 4);
    nn::Parameter y_pred(nn::DT_FLOAT, 2 * 4);
    nn::Parameter grad_y(nn::DT_FLOAT, 2 * 4);
    nn::Parameter grad_x(nn::DT_FLOAT, 2 * 4);
    
    // Fill with training data
    auto x_train_span = x_train.span<float>();
    auto y_true_span = y_true.span<float>();
    for (int i = 0; i < 8; i++) {
        x_train_span[i] = i * 0.1f;
        // Simple target: identity function
        y_true_span[i] = x_train_span[i];
    }
    
    // Get parameters for SGD
    std::vector<nn::Parameter*> params;
    network.Parameters(&params);
    
    // Training hyperparameters
    const float learning_rate = 0.01f;
    const int epochs = 10;
    
    std::cout << "Training FastFeedforward network for " << epochs << " epochs\n";
    std::cout << "Learning rate: " << learning_rate << "\n";
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        network.Forward(x_train.const_matrix<float>(2, 4), y_pred.matrix<float>(2, 4));
        
        // Calculate MSE loss
        float loss = 0.0f;
        auto y_pred_span = y_pred.span<float>();
        auto grad_y_span = grad_y.span<float>();
        
        for (int i = 0; i < 8; i++) {
            float error = y_pred_span[i] - y_true_span[i];
            loss += error * error;
            // Gradient of MSE loss
            grad_y_span[i] = 2.0f * error;
        }
        loss /= 8.0f;
        
        // Backward pass
        network.Backward(x_train.const_matrix<float>(2, 4),
                         grad_y.const_matrix<float>(2, 4),
                         grad_x.matrix<float>(2, 4));
        
        // Update parameters with SGD
        for (auto param : params) {
            if (param->HasGradient()) {
                auto weights = param->span<float>();
                auto grads = param->gradient_span<float>();
                for (size_t i = 0; i < param->size(); i++) {
                    weights[i] -= learning_rate * grads[i];
                }
                // Zero gradients for next iteration
                param->ZeroGradient();
            }
        }
        
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss << "\n";
        
        // Loss should decrease over time
        if (epoch > 0) {
            EXPECT_TRUE(std::isfinite(loss)) << "Loss is not finite at epoch " << epoch + 1;
        }
    }
    
    // Verify final predictions are closer to targets
    network.Forward(x_train.const_matrix<float>(2, 4), y_pred.matrix<float>(2, 4));
    float final_loss = 0.0f;
    auto y_pred_span = y_pred.span<float>();
    for (int i = 0; i < 8; i++) {
        float error = y_pred_span[i] - y_true_span[i];
        final_loss += error * error;
    }
    final_loss /= 8.0f;
    
    std::cout << "Final loss: " << final_loss << "\n";
    
    // The network should have learned something (loss decreased)
    EXPECT_TRUE(final_loss < 1.0f) << "Final loss too high: " << final_loss;
}