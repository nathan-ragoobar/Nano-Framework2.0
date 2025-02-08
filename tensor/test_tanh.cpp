#include "fixed_point.hpp"
#include <fstream>
#include <iomanip>
#include "fpm/fpm.hpp"

int main() {
    std::ofstream outFile("tanh_results.csv");
    outFile << "input,fixed_point_tanh,std_tanh\n";
    
    for(float x = -10.0f; x <= 10.0f; x += 0.01f) {
        fpm::fixed_16_16 fp(x);
        //fixed_point_7pt8 fp(x);
        //float fp_tanh = tanh(fp).to_float();
        float fp_tanh = float(tanh(fp));
        //float std_tanh = std::tanh(x);
        
        outFile << std::fixed << std::setprecision(6)
                << x << "," 
                << fp_tanh << "\n";
                //<< std_tanh << 
    }
    
    outFile.close();
    return 0;
}