# PyTorch Distributed Training

## Prerequisite

See [requirements.txt](requirements.txt)

## Files

- test_data_parallelism.py
    - Nearly identical to `Accelerate`'s [example](https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py) but using a larger model and changing the default batch_size settings.
    - Launch dual GPU training on single node with mixed precision:
        ```
        python -m torch.distributed.run --nproc_per_node 2 --use_env test_data_parallelism.py --fp16=True
        ```
- test_model_parallelism.py
    - Data parallelism & model parallelism without relying on `Accelerate` library.
    - Only supports single node dual GPU training without mixed precision.
    - Launch dual GPU training on single node without mixed precision:
        ```
        python test_model_parallelism.py 
        ```

## License

Apache License 2.0

