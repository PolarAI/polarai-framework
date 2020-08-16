//===----------------------------------------------------------------------===//
// Copyright (c) 2020 PolarAI. All rights reserved.
//
// Licensed under MIT license.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/IR/Attributes.h"
using namespace clang;

namespace {

struct KernelAttrInfo : public ParsedAttrInfo {
  KernelAttrInfo() {
    OptArgs = 0;
    static constexpr Spelling S[] = {{ParsedAttr::AS_GNU, "kernel"},
                                     {ParsedAttr::AS_CXX11, "kernel"},
                                     {ParsedAttr::AS_CXX11, "polarai::kernel"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema& S, const ParsedAttr& Attr,
                            const Decl* D) const override {
    // This attribute appertains to functions only.
    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << Attr << "functions";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(Sema& S, Decl* D,
                                   const ParsedAttr& Attr) const override {
    // Check if the decl is at file scope.
    if (!D->getDeclContext()->isFileContext()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          DiagnosticsEngine::Error,
          "'kernel' attribute only allowed at file scope");
      S.Diag(Attr.getLoc(), ID);
      return AttributeNotApplied;
    }
    // Check if we have an optional string argument.
    if (Attr.getNumArgs() > 0) {
      return AttributeNotApplied;
    }
    // Attach an annotate attribute to the Decl.
    D->addAttr(AnnotateAttr::Create(S.Context, "kernel", Attr.getRange()));
    return AttributeApplied;
  }
};

} // namespace

static ParsedAttrInfoRegistry::Add<KernelAttrInfo> X("kernel", "");
