from medium_trading.config import Settings


def main() -> None:
    settings = Settings()
    symbols = ", ".join(settings.symbols)
    print("medium-trading bootstrap ready")
    print(f"symbols: {symbols}")
    print(f"timeframes: {', '.join(settings.timeframes)}")
    print("strategy: trend_pullback")


if __name__ == "__main__":
    main()
