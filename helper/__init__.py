from flask import Flask, render_template

render = render_template

def view(file_name):
  return file_name + '.html'