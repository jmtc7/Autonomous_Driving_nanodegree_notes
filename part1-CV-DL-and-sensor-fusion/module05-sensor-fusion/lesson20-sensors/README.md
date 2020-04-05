# Sensors

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

So far we have worked a lot with cameras, which are really important and useful but, same as humans have a nose because our eyes can not smell and ears because our noses can not hear, we need LIDARs to measure the distances (which can not be done with a normal camera) and RADARs to see through a dense fog that may limitate the LIDAR's range. In self-driving cars, we usually get information from different sensors and fuse it.

By the end of this module, I will implement a data fusion algorithm to track a pedestrial relative to a car using LIDAR and RADAR data using the Kalman Filter.


## Used sensors in autonomous vehicles
Mercedes have an autonomous van called chewbacca that has over 15 sensors, including 3D stereo cameras, monocular cameras with special lenses to increase their range to properly classify traffic signs far from the car, RADARs and LIDARs.

### RADARs
RADARs are usually used near the ground to measure distances. They can use the Doppler Effect to measure speed directly instead of use the position variation measuring the distance to an object twice. They can also be used to create maps getting measurements to transparent (but hard) surfaces that a LIDAR will not be able to consider. It is also the least affected by rain and fog and it has a big field of view (around 150º) and range (>200 m).

However, they have a reduced resolution (compared to LIDARs and, specially, cameras), which makes noise and obstructions a bigger problem

### LIDARs
Most of them use infrarred lasers (around 900 nm), but some of them use longer wavelengths to increase their range and performance in rainy or foggy conditions. Their resolution is higher than the RADAR's because the laser beams are more focused and they usually provide several vertical layers with more point in each of them.

They can not measure the velocity directly, they need to compare two distance measurements. They are also more affected by weather conditions and sensor dirtiness. They are also bigger than cameras and radars, so they are more difficult to integrate in a subtle way.
