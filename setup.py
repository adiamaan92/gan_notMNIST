import logging
import subprocess
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install


class CustomCommands(install):

  def run_custom_command(self, command_list):
    print 'Running command: %s' % command_list
    p = subprocess.Popen(
        command_list,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    stdout_data, _ = p.communicate()
    print 'Command output: %s' % stdout_data
    logging.info('Log command output: %s', stdout_data)
    if p.returncode != 0:
      raise RuntimeError('Command %s failed: exit code: %s' %
                         (command_list, p.returncode))

  def run(self):
    self.run_custom_command(['apt-get', 'update', '--force-yes'])
    self.run_custom_command(
          ['sudo', 'apt-get', 'install', '-y', 'python-tk'])

    install.run(self)
REQUIRED_PACKAGES = ['matplotlib>=1.5.3', 'keras']

setup(
    name='notMNIST',
    version='0.1',
    author = 'Adiamaan Keerthi',
    author_email = 'mak.adi55@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Trying notMNIST on GCP',
    requires=[]
    )
