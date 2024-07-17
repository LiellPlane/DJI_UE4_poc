using maturin and pyo3 for calling rust from python

maturin new {name} creates all the gobbledegook 

maturin needs to be pip installed into a virtual env or it will cry

maturin develop --release will then build the current maturin project and plop the binary somewhere in the virtual env to be callable as a standard python module 