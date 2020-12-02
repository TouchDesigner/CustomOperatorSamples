# FilterDAT

Thisi s a simple examlpe of a Custom Filter DAT operator.
It's functionality is to convert input strings into `Upper Camel Case`, `Upper Case` or `Lower Case` as well as the possibility to strip spaces from the input string.

## Parameters
* **Case** - select how to convert the input string
  * _Upper Camel Case_ - will convert the string to Upper Camel Case. For example `ARUCO Markers` will become `Aruco Markers`
  * _Lower Case_ - will convert the string to Lower Case. For example `ARUCO Markers` will become `aruco markers`
  * _Upper Case_ - will convert the string to Upper Case. For example `ARUCO Markers` will become `ARUCO MARKERS`
* **Keep Spaces** - When enabled will preserve spaces in the input string, otherwise will strip spaces.
