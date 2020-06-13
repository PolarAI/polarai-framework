#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <athena/backend/llvm/runtime/Event.h>

namespace athena::backend::llvm {
class OpenCLEvent : public Event {
public:
  explicit OpenCLEvent(cl_event evt) : mEvent(evt){};
  
  auto getNativeEvent() -> cl_event& {
    return mEvent;
  }

private:
  cl_event mEvent;
};
}
