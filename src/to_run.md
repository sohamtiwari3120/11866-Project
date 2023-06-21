0. default without text
out (1024, 64, 56)
`overall l2: 34.991707`

# With concat (audio+text)
1. **w text full**
out (1024, 64, 56)
`overall l2: 35.42975`
rerun
out (1024, 64, 56)
`overall l2: 35.464024`

2. **w text aligned**
out (1024, 64, 56)
`overall l2: 35.69481`

3. **w text segmented**
out (1024, 64, 56)
`overall l2: 35.301964`

# W/O Concat
4. **w text full**
out (1024, 64, 56)
`overall l2: 35.318687`
rerun
out (1024, 64, 56)
`overall l2: 35.03264`

5. **w text aligned**
out (1024, 64, 56)
`overall l2: 34.93044`

6. **w text segmented**
out (1024, 64, 56)
`overall l2: 34.79656`

# With concat (text+audio)
7. **w text full**
out (1024, 64, 56)
`overall l2: 35.723812`

8. **w text aligned**
out (1024, 64, 56)
`overall l2: 35.81446`

9. **w text segmented**
out (1024, 64, 56)
`overall l2: 35.491974`

