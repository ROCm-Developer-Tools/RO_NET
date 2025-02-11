"""
******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************
"""

import os

types = [
    ("float", "float"),
    ("double", "double"),
    ("char", "char"),
    ("signed char", "schar"),
    ("short", "short"),
    ("int", "int"),
    ("long", "long"),
    ("long long", "longlong"),
    ("unsigned char", "uchar"),
    ("unsigned short", "ushort"),
    ("unsigned int", "uint"),
    ("unsigned long", "ulong"),
    ("unsigned long long", "ulonglong"),
]


def put_api_x(GRAN, T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_put_{GRAN}(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_put_{GRAN}(\n"
        f"    {T} *dest, const {T} *source, size_t nelems, int pe);\n\n"
    )


def generate_put_api_x():
    expanded_code = """
/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in a wave must collectively participate
 * in the call using the same arguments
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += put_api_x("wave", type_, tname_)
    
    expanded_code += """
/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into rocshmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * (WG) granularity. However, All threads in a WG must collectively participate
 * in the call using the same arguments.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += put_api_x("wg", type_, tname_)

    return expanded_code


def get_api_x(GRAN, T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_get_{GRAN}(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_get_{GRAN}(\n"
        f"    {T} *dest, const {T} *source, size_t nelems, int pe);\n\n"
    )


def generate_get_api_x():
    expanded_code = """
/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However,  all threads in the wave must participate in the
 * call using the same parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += get_api_x("wave", type_, tname_)
    
    expanded_code += """
/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However,  all threads in the workgroup must participate in
 * the call using the same parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += get_api_x("wg", type_, tname_)

    return expanded_code


def put_nbi_api_x(GRAN, T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_put_nbi_{GRAN}(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_put_nbi_{GRAN}(\n"
        f"    {T} *dest, const {T} *source, size_t nelems, int pe);\n\n"
    )    


def generate_put_nbi_api_x():
    expanded_code = """
/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in the wave must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += put_nbi_api_x("wave", type_, tname_)

    expanded_code += """
/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in the WG must call in with the sameo
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += put_nbi_api_x("wg", type_, tname_)

    return expanded_code


def get_nbi_api_x(GRAN, T, TNAME):
    return (
        f"__device__ ATTR_NO_INLINE void rocshmem_ctx_{TNAME}_get_nbi_{GRAN}(\n"
        f"    rocshmem_ctx_t ctx, {T} *dest, const {T} *source,\n"
        f"    size_t nelems, int pe);\n"
        f"__device__ ATTR_NO_INLINE void rocshmem_{TNAME}_get_nbi_{GRAN}(\n"
        f"    {T} *dest, const {T} *source, size_t nelems, int pe);\n\n"
    )


def generate_get_nbi_api_x():
    expanded_code = """
/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in the wave must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += get_nbi_api_x("wave", type_, tname_)

    expanded_code += """
/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * rocshmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in the WG must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */\n"""
    for type_, tname_ in types:
        expanded_code += get_nbi_api_x("wg", type_, tname_)

    return expanded_code


def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)


def generate_RMA_X_header(output_dir, copyright):
    expanded_code = copyright

    expanded_code += """
#ifndef LIBRARY_INCLUDE_ROCSHMEM_RMA_X_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_RMA_X_HPP

namespace rocshmem {
"""

    expanded_code += (
        generate_put_api_x() +
        generate_get_api_x() +
        generate_put_nbi_api_x() +
        generate_get_nbi_api_x()
    )

    expanded_code += """
}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_RMA_X_HPP
"""

    output_file = os.path.join(
        output_dir, 'rocshmem_RMA_X.hpp'
    )

    write_to_file(output_file, expanded_code)
