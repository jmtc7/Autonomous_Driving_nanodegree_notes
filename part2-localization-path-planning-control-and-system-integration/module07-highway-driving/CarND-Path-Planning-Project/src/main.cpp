#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"

// ADDED to generate trajectories. Library from https://kluge.in-chemnitz.de/opensource/spline/
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main()
{
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;


  //********************************************************
  // STEP 1: OPEN FILE WITH THE WAYPOINTS LIST AND READ THEM
  //********************************************************

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line))
  {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }


  //*************************************
  // STEP 2 [DONE]: SETUP SOME PARAMETERS
  //*************************************

  // Lane in which the ego car will drive (it will be modified by the ones where the waypoints are)
  //// 0 left, 1 middle, 2 right
  int lane = 1; 

  // Target velocity (MPH)
  //// At which the car will try to drive
  //// Slightly below the speed limit
  double ref_vel = 0.0;


  
  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy,&lane,&ref_vel]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode)
  {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry")
        {


          //****************************************
          // STEP 3: GET EGO CAR'S LOCALIZATION DATA
          //****************************************
          //// i.e. Frenet coords, car angle and speed, etc
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];


          // DONE: Get how many points were not executed by the previous trajectory
          int prev_traj_size = previous_path_x.size();


          //*****************************************************
          // STEP 4: GET INFO FROM OTHER VEHICLES (SENSOR FUSION)
          //*****************************************************

          // List of all other cars on the same side of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          // If there are any unprocessed points, set the "s" of the state to the last of them
          if(prev_traj_size > 0)
            car_s = end_path_s;

          // Environment flags
          bool car_ahead = false; // If the ego car has a car in front of it that is too close
          bool car_left = false; // If there is a car on the left lane (preventing us from changing to it)
          bool car_right = false;  // If there is a car on the right lane (preventing us from changing)

          // Find near cars in front of the ego car which are in the same lane
          //// Iterate through all the detected cars
          //// Each "sensor_fusion" element is this info of one car: [id, x, y, vx, vy, s, d]
          for(int i=0; i<sensor_fusion.size(); i++)
          {
            // Get analysed car's "d"
            double d = sensor_fusion[i][6];

            // Get in which lane the analysed car is driving
            int anal_car_lane = -1;

            if(d>0 and d<4)
              anal_car_lane = 0;
            else if(d>4 and d<8)
              anal_car_lane = 1;
            else if(d>8 and d<12)
              anal_car_lane = 2;
            else
              continue; // If the car is not in the right side of the road, pass to the next car
            

            // If the car is in the right side of the road (i.e. driving in the same direction as the ego car)
            // Get analysed car's velocity
            double vel_x = sensor_fusion[i][3];
            double vel_y = sensor_fusion[i][4];
            double mag_vel = sqrt(pow(vel_x,2) + pow(vel_y,2)); // Magnitude of the velocity

            // Get analysed car's "s"
            double s = sensor_fusion[i][5];

            // Project the "s" value further away
            //// If we are using points from the previous path
            //// The ego car has observed something now, but it is not there yet, it has to finish the previous path before
            //// A constant "mag_vel" is assumed to be constant for all the points on the previous trajectory
            s += (double)prev_traj_size*mag_vel*0.02;

            // Set flags depending on which lane the analysed car is driving on
            //// Danger is considered at 30 m ahead and +-15 m ahead or behind for lane changes
            if(anal_car_lane == lane)
              car_ahead |= s>car_s and s<(car_s+30.0);
            else if(anal_car_lane == (lane-1))
              car_left |= s>(car_s-30.0) and s<(car_s+30.0);
            else if(anal_car_lane == (lane+1))
              car_right |= s>(car_s-30.0) and s<(car_s+30.0);
          }



          //****************************************************
          // STEP 5 [DONE]: CREATE THE TRAJECTORY TO BE FOLLOWED
          //****************************************************

          json msgJson;

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          // Decide what to do if the ego car is following a slow car 
          double delta_vel = 0.0; // Velocity increment to be applied afterwards
          const double MAX_SPEED = 49.5;
          const double MAX_ACC = 0.224; // 5m/s^2 applied in 0.02s are a 0.1m/s variation in the velocity, which is 0.224 mph
          
          // If there is a car near the ego car ahead of it
          if (car_ahead)
          { 
            std::cout << "[!] WARNING! The car in front is too close!" << std::endl;
            std::cout << "  [i] Decision: "; 
            // If we can pass it moving to the left lane
            if (!car_left and lane>0)
            {
              std::cout << "Change to the left lane" << std::endl;
              lane--; // Change lane left.
            }
            // If we can pass it moving to the right lane
            else if (!car_right and lane!=2)
            {
              std::cout << "Change to the right lane" << std::endl;
              lane++; // Change lane right.
            }
            // If it is not possible ot change lane
            else
            {
              std::cout << "Slow down" << std::endl;
              delta_vel -= MAX_ACC; // Reduce velocity
            }
          }
          // If there is no car in front (and too close) of the ego car
          else 
          {
            // If the ego car is not in the middle lane
            if (lane != 1) 
            {
              // If it is possible to change to the middle lane
              if ((lane==0 and !car_right) or (lane==2 and !car_left)) 
                lane = 1; // Back to middle lane.
            }

            // If current velocity is below the maximum allowed
            if (ref_vel < MAX_SPEED) 
              delta_vel += MAX_ACC; // Increase velocity
          }


          // Use an spline to generate a smooth trajectory (avoid high perks)
          // Vectors to store the trajectory points 
          std::vector<double> pts_x;
          std::vector<double> pts_y;

          // Define variables to define the last car state
          //// Current state is given by car_x, car_y and car_yaw (meters and degrees)
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);

          // Use two last points of the previous trajectory
          // If we do not have two points available -> Use the current state and approximate another one
          if(prev_traj_size < 2)
          {
            //Generate an approximation of the point before the current state
            double prev_car_x = car_x - cos(car_yaw);
            double prev_car_y = car_y - sin(car_yaw);

            // Push the created point 
            pts_x.push_back(prev_car_x);
            pts_y.push_back(prev_car_y);

            // Push current state
            pts_x.push_back(car_x);
            pts_y.push_back(car_y);
          }
          else
          {
            // Set the reference position to the last non-processed point
            ref_x = previous_path_x[prev_traj_size-1];
            ref_y = previous_path_y[prev_traj_size-1];

            // Push the point before the last non-processed one
            double ref_x_prev = previous_path_x[prev_traj_size-2];
            double ref_y_prev = previous_path_y[prev_traj_size-2];
            ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);

            // Push the state before the last one
            pts_x.push_back(ref_x_prev);
            pts_y.push_back(ref_y_prev);

            // Push last state
            pts_x.push_back(ref_x);
            pts_y.push_back(ref_y);
          }

          // Push 3 sparse points
          // Set the d Frenet dimension to the center of the configured desired lane
          //// each lane is 4 meters wide (+2 to be in the center)
          double lane_d = 4*lane + 2;

          //// 30, 60 and 90 meters in the s Frenet dimension
          //// Need conversion from Frenet to cartesian coordinates
          std::vector<double> next_xy1 = getXY(car_s+30, lane_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          std::vector<double> next_xy2 = getXY(car_s+60, lane_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
          std::vector<double> next_xy3 = getXY(car_s+90, lane_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);

          // Push converted points
          pts_x.push_back(next_xy1[0]);
          pts_x.push_back(next_xy2[0]);
          pts_x.push_back(next_xy3[0]);

          pts_y.push_back(next_xy1[1]);
          pts_y.push_back(next_xy2[1]);
          pts_y.push_back(next_xy3[1]);

          // NOTE: So far, pts_x and pts_y contain 5 points, the two from the "last" trajectory and the new sparse ones
          //// The last trajectory ones are used to get smooth transitions between trajectories

          // Shift car reference to 0 degrees
          //// To avoid problems with having several Y values for each X value
          //// While using splines
          //// THE RESULT: the first point of the trajectory will be at [0; 0] with 0 angle respect the X axis
          for(int i=0; i<pts_x.size(); i++)
          {
            // Computed shifted coordinates
            double shifted_x = pts_x[i] - ref_x;
            double shifted_y = pts_y[i] - ref_y;

            // Correct the point in the vector
            pts_x[i] = shifted_x*cos(0-ref_yaw) - shifted_y*sin(0-ref_yaw);
            pts_y[i] = shifted_x*sin(0-ref_yaw) + shifted_y*cos(0-ref_yaw);
          }

          // Create spline
          tk::spline s;

          // Compute spline to fit our point collection
          s.set_points(pts_x, pts_y);

          // Push the points that were not processed during the last time step
          //// These will be processed in the next timestep (with the ones created now)
          for (int i = 0; i<prev_traj_size; i++)
          {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          // Compute how much we need to advance in each time step
          //// In order to keep the speed below the limit one ("ref_vel" variable)
          //// Divide the segment connecting the car and the spline 30m in front of us and uniformly divide it
          //// We use 0.02 because they are the seconds of each time step, the limit velocity ("ref_vel") and the connexion distance
          double target_x = 30.0;
          double target_y = s(target_x); //spline value fot x = our horizon (e.g. 30 m)
          double distance = sqrt(pow(target_x,2) + pow(target_y,2));
          

          // Generate as much points as necessary to reach 50 points
          //// The last trajectory may have some points that have not been processed
          double x_add_on = 0; // X offset to generate points
          for(int i=1; i<50-prev_traj_size; i++)
          {
            // Variate velocity according to the decissions before
            ref_vel += delta_vel;
            if (ref_vel > MAX_SPEED)
              ref_vel = MAX_SPEED;
            else if (ref_vel < MAX_ACC)
              ref_vel = MAX_ACC;

            // Compute how much we need to advance in each time step
            double advance_per_point = distance/(0.02*ref_vel/2.24); //2.24 to convert from MPH to m/s

            // Build point using spline
            double x_point = x_add_on + target_x/advance_per_point;
            double y_point = s(x_point);

            // Update x offset and "last point" references
            x_add_on = x_point;

            // Helper variables in order not to use modified values
            double x_point_cp = x_point;
            double y_point_cp = y_point;

            // Undo the transformation to "simulate" that we have the car at 0 degrees
            x_point = x_point_cp*cos(ref_yaw) - y_point_cp*sin(ref_yaw);
            y_point = x_point_cp*sin(ref_yaw) + y_point_cp*cos(ref_yaw);

            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }


          //***********************************************
          // STEP 6: ADD TRAJECTORY TO THE MESSAGE AND SEND
          //***********************************************

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req)
  {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length)
  {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}
