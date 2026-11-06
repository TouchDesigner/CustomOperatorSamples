# Basic Generator CHOP

The Basic Generator CHOP is a barebones example for creating a custom CHOP generator. The example creates a number of channels with certain length. 
The channel values are acted upon depended by the operation choosen and their index. For example with the operation `Add` the first channel has 0 added (the first channel's index is 0) while the second channel has 1 added (the second channel's index is 1).

## Parameters
* **Length** - The number of samples in the operator.
* **Number of Channeles** - The number of channels created by the operator.
* **Apply Scale** - When enabled, the channel values are multiplied by the value of the `Scale` parameter.
* **Scale** - The value the channels are multiplied by.
* **Operation** - The Operation that is done on the created channels depending on their channel index.
  * **Add** - Each channel has its index added to it. 
  * **Multiply** - Each channel is multiplied by its index.
  * **Power** - Each channel is taken to the power of its index.

## Python

The example also shows how a Custom Operator can extend its node with Python, using the CPython API. It is built against the Python that ships with TouchDesigner (see [Using TouchDesigner's Python](../../README.md#using-touchdesigners-python)).

These members are added to the node's Python class:
* **Offset** - Read/write. A value added to every sample, e.g. `op('basicgenerator1').Offset = 0.5`.
* **ExecuteCount** - Read-only. The number of times the node has cooked.
* **Reset()** - Sets `Offset` back to 0.

The node also gets a callbacks DAT with a `getScale(op, scale)` function, which is called on every cook with the scale about to be used (1.0 when Apply Scale is off) and can return a different one.

The Python members and the callbacks DAT are only available when the plugin is installed as a Custom Operator, not when it is loaded into a CPlusPlus CHOP.
