# Wrap Points SOP

This example implements a SOP which uses sendRay to wrap input 1 to input 2.

## Parameters
* **Rays**:	One of [Parallel, Radial], which determines in which way the rays are sent.
* **Direction**: If Rays is Parallel, it will send rays from the points of input 1 in this vector's direction.
* **Destination**: If Rays is Radial, it will send rays from the points of input 1 towards this destination.
* **Reverse**: If On, reverses the direction in which the rays are sent.
* **Hit Color**: Which color to set those points that hit input 2.
* **Miss Color**: Which color to set those points that did not hit input 2.
* **Scale**: A scale for hitLength, 0 a hit point will stay in its original position; 1 a hit point will be at the location
	where it hit the second input.

	This SOP takes two inputs and wraps the first one onto the second one.