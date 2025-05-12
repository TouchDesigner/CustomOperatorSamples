# Sprinkle SOP
This is a proof of concept version of the now builtin TouchDesigner Sprinkle SOP. The Sprinkle SOP is used to add points to either the surface or the volume of a SOP. You can create points on a surface, or within a closed volume based on the Method menu. The Surface method keeps the distribution of points constant per unit area of surface, whereas Per-Primitive gives each primitive, usually triangles, a constant number per primitive no matter their size.

## Parameters
* **Seed** - Any number, integer or non-integer, which starts the random number generator. Each number gives completely different point positions.
* **Generate** - Describes where points are located.
  * **Surface Area** - Points are evenly distributed along the entire surface of the geometry. That is, smaller polygons receives fewer points on average than larger polygons, etc.
  * **Per Primitive** - Points are distributed equally amongst each primitive. That is, each polygon receives the same number of points.
  * **Bounding Box** - Points are distributed within the bounding box volume of the geometry. This can be faster, though less accurate than the volume method.
  * **Inside Volume** - Points are evenly distributed within the entire volume of the geometry.
* **Point Count** - The total number of points created.
* **Separate Points** - Remove points until remaining are a minimum distance apart.
* **Minimum Distance** - Minimum distance when using Consolidate option.
