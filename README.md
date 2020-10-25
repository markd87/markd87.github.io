# My Personal Website and Blog

![Update and Publish Blog](https://github.com/markd87/markd87.github.io/workflows/Update%20and%20Publish%20Blog/badge.svg)

The website is built using GitHub pages and Jekyll.

The repo uses GitHub actions to trigger a build of the site and push it to the gh-pages branch from which the website is published.
Some of the posts are generated using [fastpages](https://github.com/fastai/fastpages) which allows to convert Jupyter notebooks to Jekyll posts.
A scheduled action is used to run periodic updates of the dynamic notebooks that have time dependent input.
