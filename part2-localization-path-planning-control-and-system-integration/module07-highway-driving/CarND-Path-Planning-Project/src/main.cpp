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


  //*******************************************************
  // STEP 1: Open file with list of waypoints and read them
  //*******************************************************
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
  double ref_vel = 49.5;


  
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
          //**********************************
          // STEP 3: Get our localization data
          //**********************************
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
          // STEP 4: Get info from other vehicles (sensor fusion)
          //*****************************************************

          // List of all other cars on the same side of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];


          //**************************************************
          // STEP 5 [DONE]: Define a trajectory to be followed
          //**************************************************

          json msgJson;

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          /*
          // DONE: 1st trial - Drive in an straight line with constant vel (xy coords)
          double dist_inc = 0.5;
          for (int i = 0; i < 50; i++)
          {
            next_x_vals.push_back(car_x+(dist_inc*i)*cos(deg2rad(car_yaw)));
            next_y_vals.push_back(car_y+(dist_inc*i)*sin(deg2rad(car_yaw)));
          }
          */

         /*
          // DONE: 2nd trial - Drive in the ego car's lane using Frenet Coordinates and slower
          double dist_inc = 0.33; // Advance less each time step to go slower
          for (int i = 0; i < 50; i++)
          {
            // Increase S the given amount
            double next_s = car_s + ((i+1)*dist_inc); //i+1 so our first point is different to the last car state
            // Keep D
            double next_d = car_d;

            // Convert from SD to XY
            std::vector<double> next_xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);

            next_x_vals.push_back(next_xy[0]);
            next_y_vals.push_back(next_xy[1]);
          }
          */

          // DONE: 3rd trial - Use an spline to generate a smooth trajectory (avoid high perks)
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
            //Generate an aproximation of the point before the current state
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
          for (int i = 0; i<previous_path_x.size(); i++)
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
          double advance_per_point = distance/(0.02*ref_vel/2.24); //2.24 to convert from MPH to m/s

          // Generate as much points as necessary to reach 50 points
          //// The last trajectory may have some points that have not been processed
          double x_add_on = 0; // X offset to generate points
          for(int i=1; i<50-previous_path_x.size(); i++)
          {
            // Build point using spline
            double x_point = x_add_on + target_x/advance_per_point;
            double y_point = s(x_point);

            // Update x offset and "last point" references
            x_add_on = x_point;

            // Undo the transformation to "simulate" that we have the car at 0 degrees
            x_point = x_point*cos(ref_yaw) - y_point*sin(ref_yaw);
            y_point = x_point*sin(ref_yaw) + y_point*cos(ref_yaw);

            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }





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