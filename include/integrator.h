#include "eikonal.h"

namespace drrt {

template <bool ad>
class tracer {
public:

	tracer(Float<false> ds);

	tracer();

	void step();

protected:
	~tracer();



	Float<false> ds;
};

} // namespace drrt