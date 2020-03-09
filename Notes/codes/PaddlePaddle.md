# Pieces of Codes during Coding

## Paddle

### Get parameters according to var names

Basic methods:

``` python
var_name = ""

params = np.array(fluid.global_scope().find_var(var_name).get_tensor())

```

There is also a `set_tensor()`, but don't know how to use for now.

Advanced usage:

* Use a dict to store that var:
``` python
params_[var_name] = params
```

* Find all vars in a program:
``` python
for var in program.list_vars():
    var_name = var.name
    params = np.array(fluid.global_scope().find_var(var_name).get_tensor())
    params_[var_name] = params
```

<strong>Note</strong>: NOTE that all parameters must be got after `exe.run()` is called. Otherwise, `find_var(var_name)` will return a `None` type. 
