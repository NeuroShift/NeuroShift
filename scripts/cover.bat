@echo off

coverage run -m pytest --ignore=./tests/view/pages
coverage html