# smath

## Usage
Using smath is as simple as few lines of code. The only things we have to do, is to import the library and create two functions.
One of the functions would be the x-axis of the new system of reference, and the other would be defined based on it in the new system:
```python
>>> from smath import plot, relfunc
>>> plot(lambda x: 2*x, lambda x: x + 2)
```
