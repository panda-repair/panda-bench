"""Compatibility for older pip editable installs, without runtime metadata imports."""
import sys
from setuptools import setup
from setuptools.command.develop import develop


class DirectDevelop(develop):
    def install_script(self, dist, script_name, script_text, dev_path=None):
        if script_name == "panda-bench":
            # Legacy develop wraps script-files with pkg_resources, importing
            # socket before our CLI starts. Install the shipped script directly.
            lines = script_text.split("\n")
            if lines[0].startswith("#!"):
                lines = lines[1:]
            text = "#!" + sys.executable + "\n" + "\n".join(lines)
            self.write_script(script_name, text.encode("utf-8"), "b")
        else:
            super().install_script(dist, script_name, script_text, dev_path)


setup(cmdclass={"develop": DirectDevelop})
