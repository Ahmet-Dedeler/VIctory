ByteAddressBuffer source : register(t0);
RWByteAddressBuffer writable : register(u0);
RWByteAddressBuffer observed : register(u1);
cbuffer Arguments : register(b0) { uint index; }

[numthreads(1, 1, 1)]
void verify() {
    uint input = source.Load(0);
    uint before = writable.Load(0);
    writable.Store(0, before ^ input);
    observed.Store4(index * 16, uint4(input, source.Load(4), before, writable.Load(0)));
}
