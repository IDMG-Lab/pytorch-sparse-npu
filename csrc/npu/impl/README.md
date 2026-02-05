## 如何加入一个算子

  在 `csrc/ops_.../` 下新建对应算子目录，例如 `op_name = a`：

1. 新建目录结构  
   - `csrc/ops_.../a/op_host`  
   - `csrc/ops_.../a/op_kernel`

2. 在上述目录中添加对应实现代码  
   - Host 侧实现放在 `op_host`  
   - Kernel 侧实现放在 `op_kernel`

3. 创建 `csrc/ops_.../a/CMakeLists.txt`  在其中添加：
   `register_operator(a)`

## 注意事项

1. **确认实际生成的 ACL 算子名称**  
   算子编译部署后，可在 `ops_.../build_out/autogen/` 中找到对应的自动生成 `.cpp` 文件。  
   查看最终的 **acl op_name**。

2. **核对 NPU 侧调用名称是否一致**  
   在 `npu/` 目录下定位相关函数，检查其中 `EXEC_NPU_CMD` 调用的**acl opname**。  
   - 可能存在 **大小写不一致** 的问题  
   - 必须确保 `EXEC_NPU_CMD` 中的名称与 `autogen` 生成的 **acl opname** 完全一致

<!-- ## 怎么加入算子<br>
在csrc/ops_.../ 中新建对应的算子目录 (eg.op_name=a) 
创建对应的**a/op_host**,**a/op_kernel**,添加对应的代码<br>
创建**a/CMakeLists.txt**，添加`register_operator(a)` -->

<!-- ## 一些注意事项
1.在算子编译部署之后可以通过**ops_.../build_out/autogen**中找到对应的cpp文件查看实际生成的acl算子名称<br>
2.查看npu目录下对应的函数中的**EXEC_NPU_CMD**调用算子的名称(可能有大小写的问题)，需要确保与**autogen**中的算子名称相同  
<br> -->

<!-- # 怎么添加算子重编译
1.<br>
修改对应的__init__.py，添加对应的npu分支 <br>
` npu_spec = importlib.machinery.PathFinder().find_spec(f'{library}_npu', [osp.dirname(__file__)]) `
## 1. 显示Segmation fault 或者 undefined symbol 怎么解决???<br>
重新编译部署一下算子库再进行setup install操作

# 调用方式需要使用torch.ops.torch_sparse.ind2ptr

需要修改vendorname保证两个库的算子不会被覆盖<br>
`OPP=/home/ma-user/Ascend/ascend-toolkit/latest/opp

export ASCEND_CUSTOM_OPP_PATH = $OPP/vendors/torch_scatter_npu/op_api/lib:$OPP/vendors/torch_sparse_npu/op_api/lib`  
用冒号分出不同lib path
这种方法还是没有用，无法自动获取.so文件？？？也还是会覆盖

# 两个zip文件是尝试在不同包里面进行编译算子
但是虽然最终在不同的vendor name，但是还是会冲突，无法自动获取对应的.so文件 -->