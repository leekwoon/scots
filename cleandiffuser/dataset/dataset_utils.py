def loop_dataloader(dl):
    while True:
        for b in dl:
            yield b

