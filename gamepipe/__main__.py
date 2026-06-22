from .cli import main

if __name__ == "__main__":          # guard: vLLM spawns its EngineCore by re-importing here
    main()
