# Basic Generator CHOP

The Basic Generator CHOP is a barebones example for creating a custom CHOP generator. The example creates a number of channels with certain length. 
The channel values are acted upon depended by the operation choosen and their index. For example with the operation `Add` the first channel has 0 added (the first channels index is 0) while the second channel has 1 added (the second channels index is 1).

## Parameters
* **Length** - The number of samples in the operator.
* **Number of Channeles** - The number of channels created by the operator.
* **Apply Scale** - When enabled, the channel values are multiplied by the value of the `Scale` parameter.
* **Scale** - The value the channels are multiplied by.
* **Operation** - The Operation that is done on the created channels depending on their channel index.
  * **Add** - Each channel has its index added to it. 
  * **Multiply** - Each channel is multiplied by its index.
  * **Power** - Each channel is taken to the power of its index.
