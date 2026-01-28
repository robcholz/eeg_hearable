# Optimize

```python
import cProfile
import pstats

cProfile.run("main()", "out.prof")
p = pstats.Stats("out.prof")
```

```shell
snakeviz out.prof
```