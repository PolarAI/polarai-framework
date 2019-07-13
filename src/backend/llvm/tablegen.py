import argparse

plainArgTypes = ["int", "int*", "int *"]


def get_mangled_name(name, type):
    return "athn_" + name + "_" + type[0]


def generate_wrapper(name, signature):
    defs = ""
    for template in ["float", "double"]:
        defs += "void "
        defs += get_mangled_name(name, template)
        defs += "("

        defs += "void *devicePtr, "
        defs += "void *allocatorPtr"

        arg_no = 0
        for arg in signature:
            defs += ","
            if arg not in plainArgTypes:
                defs += " void *arg" + str(arg_no) + "Ptr"
            else:
                defs += " " + arg + " arg" + str(arg_no)
            arg_no += 1

        defs += ") {\n"

        defs += "auto device = reinterpret_cast<Device*>(devicePtr);\n"
        defs += "auto allocator = reinterpret_cast<Allocator*>(allocatorPtr);\n"

        arg_no = 0
        for arg in signature:
            if arg not in plainArgTypes:
                defs += "auto arg" + str(arg_no) + " = reinterpret_cast<" + arg + ">(arg" + str(
                    arg_no) + "Ptr);\n"
            arg_no += 1
        defs += name + "<" + template + ">(device, allocator"

        arg_no = 0
        for arg in signature:
            defs += ", "
            if arg not in plainArgTypes:
                defs += "arg" + str(arg_no)
            else:
                defs += "" + arg + " arg" + str(arg_no)
            arg_no += 1
        defs += ");\n"
        defs += "}\n"
    return defs


def generate_llvm(name, signature):
    for template in ["float", "double"]:
        res = ""
        vec_name = get_mangled_name(name, template) + "_args"
        res += "std::vector<::llvm::Type *> " + vec_name + ";\n"
        res += vec_name + ".push_back(::llvm::Type::getInt64Ty(ctx));\n"
        res += vec_name + ".push_back(::llvm::Type::getInt64Ty(ctx));\n"

        for arg in signature:
            res += vec_name + ".push_back("

            if arg == "int":
                res += "::llvm::Type::getInt32Ty(ctx)"
            elif arg == "int *" or arg == "int *":
                res += "::llvm::Type::getInt32PtrTy(ctx)"
            else:
                res += "::llvm::Type::getInt64Ty(ctx)"
            res += ");\n"

        res += "::llvm::FunctionType *" + get_mangled_name(name, template) + "_FT = "
        res += "::llvm::FunctionType::get(::llvm::Type::getVoidTy(ctx), " + vec_name + ", false);\n"

        res += "::llvm::Function *" + get_mangled_name(name, template)
        res += "_F = ::llvm::Function::Create("
        res += get_mangled_name(name, template) + "_FT, "
        res += "::llvm::Function::ExternalLinkage, \""
        res += get_mangled_name(name, template) + "\", &module);\n"

        block_name = get_mangled_name(name, template) + "_block"
        res += "auto " + block_name + " = ::llvm::BasicBlock::Create(ctx, \"\", "
        res += get_mangled_name(name, template) + "_F);\n"

        res += "builder.SetInsertPoint(" + block_name + ");\n"
        res += "auto " + get_mangled_name(name, template) + "_ptr_val = "
        res += "::llvm::ConstantInt::get(::llvm::Type::getInt64Ty(ctx), "
        res += "reinterpret_cast<uint64_t>(getFunctionPtr(\""
        res += get_mangled_name(name, template) + "\")));\n"

        res += "auto " + get_mangled_name(name, template) + "_ptr = "
        res += "builder.CreateIntToPtr(" + get_mangled_name(name, template) + "_ptr_val, "
        res += get_mangled_name(name, template) + "_FT->getPointerTo());\n"

        res += "builder.CreateCall(" + get_mangled_name(name, template) + "_FT, "
        res += get_mangled_name(name, template) + "_ptr, "
        res += "getArgs(" + get_mangled_name(name, template) + "_F));\n"

        res += "builder.CreateRetVoid();\n"

        return res


def main():
    parser = argparse.ArgumentParser(
        description='Generate necessary code for Athena LLVM Runtimes.')
    parser.add_argument("inp", type=str, help="input file")
    parser.add_argument("outp", type=str, help="output file")
    parser.add_argument("mode", type=str, help="mode")

    args = parser.parse_args()

    with open(args.outp, "w") as o:
        inp = open(args.inp, "r")

        if args.mode == "wrapper":
            o.write("#include <athena/backend/llvm/device/Device.h>\n")
            o.write("#include <athena/backend/llvm/runtime/builtin.h>\n")
            o.write("#include <athena/core/inner/Tensor.h>\n")
            o.write("#include <athena/core/Allocator.h>\n")
            o.write("using namespace athena::backend::llvm;\n")
            o.write("using namespace athena::core::inner;\n\n")
            o.write("using namespace athena::core;\n\n")
            o.write("extern \"C\" {\n")
        elif args.mode == "driver":
            o.write("#include <athena/backend/llvm/runtime-driver/runtime-driver.h>\n")
            o.write("#include \"llvm/IR/Constants.h\"\n")
            o.write("#include \"llvm/IR/IRBuilder.h\"\n")
            o.write("void athena::backend::llvm::RuntimeDriver"
                    "::generateLLLVMIrBindings(::llvm::LLVMContext &ctx, ::llvm::Module &module, "
                    "::llvm::IRBuilder<> &builder) {\n")

        for line in inp:
            command = line.split(":")
            types = command[1].split(",")
            command[0] = command[0].strip()
            types = list(map(str.strip, types))

            if args.mode == "wrapper":
                o.write(generate_wrapper(command[0], types))
            elif args.mode == "driver":
                o.write(generate_llvm(command[0], types))

        if args.mode == "wrapper":
            o.write("}\n")
        if args.mode == "driver":
            o.write("}\n")


if __name__ == '__main__':
    main()
