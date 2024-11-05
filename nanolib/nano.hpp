//I'll create the nn namespace here and then do all the declarations here. That way I could use the same nn namespace in the other snippets.
//I'll do this based off the structure in the flashlight library.

#pragma once

#include "nanolib/nn/nnhead.hpp"
#include "nanolib/nn/gpt.hpp"
#include "nanolib/optim/optim.hpp"
#include "nanolib/tensor/tensor_util.hpp"
#include "nanolib/tensor/tensor_types.hpp"
#include "nanolib/utils/rand.h"
#include "nanolib/utils/utils.h"
#include "nanolib/utils/dataloader.h"
#include "nanolib/utils/tokenizer.h"