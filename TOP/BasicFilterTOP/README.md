# Filter TOP

This example implements a TOP to limit the number of colors from the input. This example
executes on CPU Memory and supports single threaded and multi threaded.

## Parameters
* **Bits per Color:**	The number of bits for the RGB channels. Therefore, if
	we set this parameter to 1. We limit our color palette to 2^(3\*1) = 8 colors.
* **Dither:**	If on, we apply a dithering algorithm to diffuse the error.
* **Multithreaded:** If on, we calculate the output for 3 frames at the same time, therefore 
	it lags from the input by 3/4 frames depending on Download Type.