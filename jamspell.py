class TSpellCorrector:
    def LoadLangModel(self, path):
        # Stub: jamspell isn't available on osx-arm64 in this environment.
        # Treat loading as successful to allow demo runs.
        return True

    def FixFragment(self, text):
        # No-op correction for stub.
        return text
