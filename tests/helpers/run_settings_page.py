import importlib

settings_page = importlib.import_module("src.pages.04_settings")

# Call the page's main() to render UI for AppTest
settings_page.main()
