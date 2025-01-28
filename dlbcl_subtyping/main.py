from .utils.cli import BaseMLCLI, BaseMLArgs


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        # This includes `--seed` param
        pass

    # Excute as: uv run main example --param 456
    class ExampleArgs(CommonArgs):
        param: int = 123

    def run_example(self, a):
        print(type(a.param), a.param)


if __name__ == '__main__':
    cli = CLI()
    cli.run()
