# Spiral SOP
This example demonstrate how to generate a more complex shape, in this case a spiral.

## Parameters
* **Orientation** - Specify which axis the spiral is being drawn around.
  * **X**
  * **Y**
  * **Z**
* **Top radius** - The radius at the top of the spiral.
* **Bottom radius** - The radius at the bottom of the spiral.
* **Height** - The length or distance between the first and last point of the spiral.
* **Turns** - The number of turns the spiral should attempt to do between the top and bottom.
* **Divisions** - The number of points or divisions making the spiral curve. Can vary based on the Output Geometry type.
* **Output Geometry** - The geometry type making the spiral.
  * **Points**
  * **Line**
  * **Triangle Strip** 
* **Strip Width** - When triangle strip Output Geomtry type is selected, it will use the Strip Width parameter as distance between two lines.
* **GPU Direct** - Load the geometry directly to the GPU.
