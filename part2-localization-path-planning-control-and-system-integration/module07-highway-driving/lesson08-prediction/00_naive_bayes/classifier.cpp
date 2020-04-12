#include "classifier.h"
#include <math.h>
#include <string>
#include <vector>

using Eigen::ArrayXd;
using std::string;
using std::vector;

// Initializes GNB
GNB::GNB()
{
  /**
   * DONE: Initialize GNB, if necessary. May depend on your implementation.
   */
   
    // "possible_labels" initialised to {"left","keep","right"} in class declaration
   
    // State's means for every class 
    left_mean = ArrayXd(4);
    left_mean << 0.0, 0.0, 0.0, 0.0;
    
    keep_mean = ArrayXd(4);
    keep_mean << 0.0, 0.0, 0.0, 0.0;
    right_mean = ArrayXd(4);
    
    right_mean << 0.0, 0.0, 0.0, 0.0;
    
    // State's variances for every class 
    left_var = ArrayXd(4);
    left_var << 0.0, 0.0, 0.0, 0.0;
    
    keep_var = ArrayXd(4);
    keep_var << 0.0, 0.0, 0.0, 0.0;
    
    right_var = ArrayXd(4);
    right_var << 0.0, 0.0, 0.0, 0.0;
    
    // State's priors for every class 
    left_prior = 0;
    keep_prior = 0;
    right_prior = 0;
}

GNB::~GNB() {}

void GNB::train(const vector<vector<double>> &data, const vector<string> &labels)
{
  /**
   * Trains the classifier with N data points and labels.
   * @param data - array of N observations
   *   - Each observation is a tuple with 4 values: s, d, s_dot and d_dot.
   *   - Example : [[3.5, 0.1, 5.9, -0.02],
   *                [8.0, -0.3, 3.0, 2.2],
   *                 ...
   *                ]
   * @param labels - array of N labels
   *   - Each label is one of "left", "keep", or "right".
   *
   * DONE: Implement the training function for your classifier.
   */
   
   // This function uses data to build a normal distribution for each variable (s, d, s_dot and d_dot)
   //// We need: mean, variance and prior (probability of each class)
   std::cout << "[i] TRAINING PROCESS STARTED" << std::endl;
   
   
   //*************************
   // STEP 0: Checks and setup
   //*************************
   std::cout << "  [i] STEP 0 started - Checks and setup" << std::endl;
   // Check that the amount of data and labels is equal
   unsigned int n_samples = data.size();
   unsigned int n_labels = labels.size();
   if(n_samples != n_labels)
   {
       std::cout << "    [!] ERROR! The number of data samples is not equal to the number of labels." << std::endl;
       std::cout << "      [i] Data samples: " << n_samples << ". Labels: " << n_labels << std::endl;
       std::cout << "      [i] Aborting GNB::train() execution..." << std::endl;
       return;
   }
   
   // Counters for the number of samples of each class
   int n_left_samples = 0;
   int n_keep_samples = 0;
   int n_right_samples = 0;
   
   
   
   //*********************
   // STEP 1: Compute mean
   //*********************
   std::cout << "  [i] STEP 1 started - Mean computation" << std::endl;
   unsigned int data_size = data[0].size();
   Eigen::ArrayXd adapted_data;
   
   // Iterate trough the data samples (and labels)
   for(unsigned int idx=0; idx<n_samples; idx++)
   {
       // Get data from this sample
       //// Use Eigen::aligned_allocator to map between std::vector and Eigenl::ArrayXd
       //// https://eigen.tuxfamily.org/dox/classEigen_1_1aligned__allocator.html
       adapted_data = Eigen::ArrayXd::Map(data[idx].data(), data_size);
       
       // Accumulate measure and count sample
       if(labels[idx] == "left")
       {
           left_mean += adapted_data;
           n_left_samples++;
       }
       else if(labels[idx] == "keep")
       {
           keep_mean += adapted_data;
           n_keep_samples++;
       }
       else if(labels[idx] == "right")
       {
           right_mean += adapted_data;
           n_right_samples++;
       }
   }
   
   // Print information about the amount of samples of each class
   std::cout << "    [i] Amount of samples per class (out of "<< n_samples << " samples):" << std::endl;
   std::cout << "      - " << n_left_samples << " left samples" << std::endl;
   std::cout << "      - " << n_keep_samples << " keep samples" << std::endl;
   std::cout << "      - " << n_right_samples << " right samples" << std::endl;
   
   // Divide accumulated measurements by the amount of samples
   left_mean /= n_left_samples;
   keep_mean /= n_keep_samples;
   right_mean /= n_right_samples;
   
   
   
   //*************************
   // STEP 2: Compute variance
   //*************************
   std::cout << "  [i] STEP 2 started - Variance computation" << std::endl;
   // Iterate trough the data samples (and labels)
   for(unsigned int idx=0; idx<n_samples; idx++)
   {
       // Get data from this sample
       //// Use Eigen::aligned_allocator to map between std::vector and Eigenl::ArrayXd
       //// https://eigen.tuxfamily.org/dox/classEigen_1_1aligned__allocator.html
       //// "data_size" declared and assigned in "STEP 1"
       adapted_data = Eigen::ArrayXd::Map(data[idx].data(), data_size);
       
       // Compute numerator
       //// Formula: https://www.mathsisfun.com/data/standard-deviation-formulas.html
       if(labels[idx] == "left")
           left_var += pow((adapted_data - left_mean), 2);
       else if(labels[idx] == "keep")
           keep_var += pow((adapted_data - keep_mean), 2);
       else if(labels[idx] == "right")
           right_var += pow((adapted_data - right_mean), 2);
   }
   
   // Square root of the means of the variances
   left_var = sqrt(left_var/n_left_samples);
   keep_var = sqrt(keep_var/n_keep_samples);
   right_var = sqrt(right_var/n_right_samples);
   
   
   
   //**********************
   // STEP 3: Compute prior
   //**********************
   std::cout << "  [i] STEP 3 started - Prior computation" << std::endl;
  
   // Normalize the amount of samples of each class
   left_prior = n_left_samples/n_samples;
   keep_prior = n_keep_samples/n_samples;
   right_prior = n_right_samples/n_samples;
  
   return;
}

string GNB::predict(const vector<double> &sample)
{
  /**
   * Once trained, this method is called and expected to return 
   *   a predicted behavior for the given observation.
   * @param observation - a 4 tuple with s, d, s_dot, d_dot.
   *   - Example: [3.5, 0.1, 8.5, -0.2]
   * @output A label representing the best guess of the classifier. Can
   *   be one of "left", "keep" or "right".
   *
   * DONE: Complete this function to return your classifier's prediction
   */
   
  //**************
  // STEP 0: setup
  //**************
  double left_prob = 0.333333;
  double keep_prob = 0.333333;
  double right_prob = 0.333333;
  
  
  //*********************************************
  // STEP 1: Compute probabilities for each class
  //*********************************************
  // For each state variable
  for(unsigned idx=0; idx<sample.size(); idx++)
  {
      // Compute Gaussian Naive Bayes's probabilities
      left_prob = (1 / sqrt(2 *M_PI*pow(left_var[idx], 2)))
                * exp(-pow(sample[idx]-left_mean[idx], 2) / (2*pow(left_var[idx], 2)));
      keep_prob = (1 / sqrt(2 *M_PI*pow(keep_var[idx], 2)))
                * exp(-pow(sample[idx]-keep_mean[idx], 2) / (2*pow(keep_var[idx], 2)));
      right_prob = (1 / sqrt(2 *M_PI*pow(right_var[idx], 2)))
                * exp(-pow(sample[idx]-right_mean[idx], 2) / (2*pow(right_var[idx], 2)));
  }
  
  // Include prior
  left_prob *= left_prior;
  keep_prob *= keep_prior;
  right_prob *= right_prior;
  
  
  //****************************************************
  // STEP 2: Search the index of the biggest probability
  //****************************************************
  // Build an std::vector with the probabilities
  std::vector<double> probs = {left_prob, keep_prob, right_prob};
  
  // Get index of the biggest element
  int max_idx = std::max_element(probs.begin(), probs.end())-probs.begin();
  
  return possible_labels[max_idx];
}
