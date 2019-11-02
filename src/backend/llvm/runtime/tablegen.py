import argparse

plainArgTypes = ["int", "int*", "int *", "float", "float *", "float*", "double", "double *",
                 "double*"]


def get_mangled_name(name, typename):
    return "athn_" + name + "_" + typename[0]


def generate_wrapper(name, signature, gentypes):
    defs = ""
    for template in gentypes:
        defs += "void "
        defs += get_mangled_name(name, template)
        defs += "("

        defs += "void *devicePtr, "
        defs += "void *allocatorPtr"

        arg_no = 0
        for arg in signature:
            defs += ","

            arg_resolved = arg

            if arg == "gentype":
                arg_resolved = template
            elif arg == "gentype *":
                arg_resolved = template + " *"
            else:
                arg_resolved = arg_resolved.replace("<gentype>", "<" + template + ">")

            if arg_resolved not in plainArgTypes:
                defs += " void *arg" + str(arg_no) + "Ptr"
            else:
                defs += " " + arg_resolved + " arg" + str(arg_no)
            arg_no += 1

        defs += ") {\n"

        defs += "auto device = reinterpret_cast<Device*>(devicePtr);\n"
        defs += "auto allocator = reinterpret_cast<Allocator*>(allocatorPtr);\n"

        arg_no = 0
        for arg in signature:
            arg_resolved = arg
            if arg == "gentype":
                arg_resolved = template
            elif arg == "gentype *":
                arg_resolved = template + " *"
            else:
                arg_resolved = arg_resolved.replace("<gentype>", "<" + template + ">")
            if arg_resolved not in plainArgTypes:
                defs += "auto arg" + str(
                    arg_no) + " = reinterpret_cast<" + arg_resolved + ">(arg" + str(
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


def generate_llvm(name, signature, gentypes):
    for template in gentypes:
        res = ""
        vec_name = get_mangled_name(name, template) + "_args"
        res += "std::vector<::llvm::Type *> " + vec_name + ";\n"
        res += vec_name + ".push_back(::llvm::Type::getInt64Ty(ctx));\n"
        res += vec_name + ".push_back(::llvm::Type::getInt64Ty(ctx));\n"

        for arg in signature:
            res += vec_name + ".push_back("

            arg_resolved = arg

            if arg == "gentype":
                arg_resolved = template
            elif arg == "gentype *":
                arg_resolved = template + " *"
            else:
                arg_resolved = arg_resolved.replace("<gentype>", "<" + template + ">")

            if arg_resolved == "int":
                res += "::llvm::Type::getInt32Ty(ctx)"
            elif arg_resolved == "int *" or arg == "int*":
                res += "::llvm::Type::getInt32PtrTy(ctx)"
            elif arg_resolved == "float":
                res += "::llvm::Type::getFloatTy(ctx)"
            elif arg_resolved == "float *" or arg == "float*":
                res += "::llvm::Type::getFloatPtrTy(ctx)"
            elif arg_resolved == "double":
                res += "::llvm::Type::getFloatTy(ctx)"
            elif arg_resolved == "double *" or arg == "double*":
                res += "::llvm::Type::getFloatPtrTy(ctx)"
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

        res += "setProperAttrs(" + get_mangled_name(name, template) + "_F);\n"

        res += "auto " + get_mangled_name(name, template) + "_fargs = "
        res += get_mangled_name(name, template) + "_F->arg_begin();\n"
        res += get_mangled_name(name, template) + "_fargs->setName(\"device\");\n"
        res += "++" + get_mangled_name(name, template) + "_fargs;\n"
        res += get_mangled_name(name, template) + "_fargs->setName(\"allocator\");\n"
        res += "++" + get_mangled_name(name, template) + "_fargs;\n"

        arg_count = 0
        for _ in signature:
            res += get_mangled_name(name, template) + "_fargs->setName(\"arg"+str(arg_count)+"\");\n"
            arg_count += 1
            res += "++" + get_mangled_name(name, template) + "_fargs;\n"

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

        res += "std::vector<::llvm::Value *> "+get_mangled_name(name, template)+"_argValues;\n"
        res += "for (auto &arg : "+get_mangled_name(name, template)+"_F->args())\n"
        res += "  " + get_mangled_name(name, template) + "_argValues.push_back(&arg);\n"

        res += "builder.CreateCall(" + get_mangled_name(name, template) + "_FT, "
        res += get_mangled_name(name, template) + "_ptr, "
        res += get_mangled_name(name, template) + "_argValues);\n"

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
            o.write("using namespace athena::backend;\n")
            o.write("using namespace athena::core::inner;\n\n")
            o.write("using namespace athena::core;\n\n")
            o.write("extern \"C\" {\n")
        elif args.mode == "driver":
            o.write("#include <runtime-driver.h>\n")
            o.write("#include \"llvm/IR/Constants.h\"\n")
            o.write("#include \"llvm/IR/IRBuilder.h\"\n")
            o.write("void athena::backend::llvm::RuntimeDriver"
                    "::generateLLVMIrBindings(::llvm::LLVMContext &ctx, ::llvm::Module &module, "
                    "::llvm::IRBuilder<> &builder) {\n")

        for line in inp:
            command = line.split(":")
            types = command[1].split(",")
            command[0] = command[0].strip()
            types = list(map(str.strip, types))

            gentypes = ["float", "double"]

            if len(command) == 3:
                if command[2].strip() != "*":
                    gentypes = command[2].split(",")
                    gentypes = list(map(str.strip, gentypes))

            if args.mode == "wrapper":
                o.write(generate_wrapper(command[0], types, gentypes))
            elif args.mode == "driver":
                o.write(generate_llvm(command[0], types, gentypes))

        if args.mode == "wrapper":
            o.write("}\n")
        if args.mode == "driver":
            o.write("}\n")


if __name__ == '__main__':
    main()
