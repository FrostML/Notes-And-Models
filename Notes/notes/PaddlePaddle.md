# Notes while learning

## InferShape

In fluid, `InferShape()` happened during compile time, while in Paddle-Lite, it's no need for any program to re-compile a model during inference. 
Hence, the value of Variables is certain and it's possible for program to get the specific value during `InferShape()`.

## memory

There are two parts of memory usage during inference. 

* One is model's parameters. 
* The other is temporary/intermediate variables. 

These two parts determine how much memory is involved during inference. 
