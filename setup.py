from setuptools import setup
setup(
  name = 'motionConstrain',         # How you named your package folder (MyLib)
  packages = ['motionConstrain'],   # Chose the same as "name"
  version = '5.0.9',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'spatial constraint (incompressible)',   # Give a short description about your library
  author = 'Yu Zheng',                   # Type in your name
  author_email = 'jorry.zhengyu@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/JorryZ/motionConstrain',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/JorryZ/motionConstrain/archive/v5.0.9.tar.gz',    # I explain this later on
  keywords = ['explicit', 'motion', 'incompressibility', 'regularization'],   # Keywords that define your package best
  install_requires=['numpy','medImgProc','scipy','trimesh','motionSegmentation'],
  classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package    
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',    
    'License :: OSI Approved :: MIT License',   # Again, pick a license    
    'Programming Language :: Python :: 3.6',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.7',
  ],
)
