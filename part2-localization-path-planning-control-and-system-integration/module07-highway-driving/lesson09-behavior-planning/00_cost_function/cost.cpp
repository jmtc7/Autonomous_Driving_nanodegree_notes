#include "cost.h"
#include <cmath>

double goal_distance_cost(int goal_lane, int intended_lane, int final_lane, double distance_to_goal)
{
  // The cost increases with both the distance of intended lane from the goal
  //   and the distance of the final lane from the goal. The cost of being out 
  //   of the goal lane also becomes larger as the vehicle approaches the goal.
 
  // Goal lane is the lane we want to be in after the whole maveuver
  // Intended lane is the one we are getting prepared to turn to
  // Final lane is the one we are going to turn tu
  // Distance to goal is delta_s (longitudinal distance) until our goal position

  /**
   * DONE: Replace cost = 0 with an appropriate cost function.
   */
   
  int delta_d = (2*goal_lane)-intended_lane-final_lane;
  double delta_s = distance_to_goal;
  
  double cost = 1 - exp(-std::abs(delta_d)/delta_s);
    
  return cost;
}
