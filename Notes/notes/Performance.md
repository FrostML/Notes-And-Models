### Python
```shell
perf stat -e cycles,stalled-cycles-frontend,stalled-cycles-backend,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,faults,minor-faults,cs,migrations -r 3 python xxx.py
```

`3` means run 3 times.
`python xxx.py` means the command.
