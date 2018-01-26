#!flask/bin/python
from flask import Flask
from flask import request
import jsonschema
import model

app = Flask(__name__)

schema = {
           "crime":  {"type" : "number"},
           "zn":     {"type" : "number"},
           "indus":  {"type" : "number"},
           "chas":   {"type" : "number"},
           "nox":    {"type" : "number"},
           "rm":     {"type" : "number"},
           "age":    {"type" : "number"},
           "dis":    {"type" : "number"},
           "rad":    {"type" : "number"},
           "tax":    {"type" : "number"},
           "ptratio":{"type" : "number"},
           "b":      {"type" : "number"},
           "lstat":  {"type" : "number"},
}

@app.route('/query', methods=['POST'])
def query():
    if not request.json:
        return 400
    try:
        jsonschema.validate(request.json, schema)
        return str(model.predict(request.json)), 200
    except(jsonschema.exceptions.ValidationError):
        return 400
    
    return 200

if __name__ == '__main__':
    app.run(debug=True)