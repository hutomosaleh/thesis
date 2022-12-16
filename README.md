# thesis

## How to use

To generate clangd lsp compile commands only:
```bash
mkdir TARGET_DIR/build
cd build && cmake -DCMAKE_COMPILE_EXPORT_COMMANDS=1 ..
```

To generate line item record data with scale factor of 0.5:
`bash generate_data.sh 0.5`

Run interactive script:
```
bash run.sh $1 $2
```
First time running after building: `bash run.sh 1 overwrite`

Set alias to ease the build and run commands:
```bash
source .aliases
build
run 1 overwrite # Run with 0:1 CPU:GPU ratio make new tpch binaries
```
