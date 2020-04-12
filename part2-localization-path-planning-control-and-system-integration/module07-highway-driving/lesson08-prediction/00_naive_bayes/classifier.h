#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include <iostream> // Added for the couts
#include "Dense"

using Eigen::ArrayXd;
using std::string;
using std::vector;

class GNB {
 public:
  /**
   * Constructor
   */
  GNB();

  /**
   * Destructor
   */
  virtual ~GNB();

  /**
   * Train classifier
   */
  void train(const vector<vector<double>> &data, const vector<string> &labels);

  /**
   * Predict with trained classifier
   */
  string predict(const vector<double> &sample);

  vector<string> possible_labels = {"left","keep","right"};
  
  // EDITED: Add containers for state means, variances and priors of each class
  //// The state is [s, d, s_dot, d_dot]
  //// The classes are "left", "keep"and "right"
  //// For each class, we will have a normal distribution and a prior for each state element
  //// NOTE: I use Eigen::ArrayXd instead of std::vector for the operations with scalars
  Eigen::ArrayXd left_mean;
  Eigen::ArrayXd keep_mean;
  Eigen::ArrayXd right_mean;
  
  Eigen::ArrayXd left_var;
  Eigen::ArrayXd keep_var;
  Eigen::ArrayXd right_var;
  
  double left_prior;
  double keep_prior;
  double right_prior;
};

#endif  // CLASSIFIER_H
