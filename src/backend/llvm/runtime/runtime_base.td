class Type;

def half : Type;
def float : Type;
def double : Type;
def void : Type;
def tensor : Type;

def ins;

class Runtime<list<Type> types = []> {
    string name = "";
    string version = "";
    string summary = "";
    string description = "";
    list<Type> supportedTypes = types;
    list<string> extraModules = [];
    list<string> extraFeatures = [];
}

class RT_Builtin<string mnemonic, list<Type> overloads = []> {
    string summary = "";
    string description = "";
    dag arguments = (ins);
    string builtinName = mnemonic;
    list<Type> overloadTypes = overloads;
}
