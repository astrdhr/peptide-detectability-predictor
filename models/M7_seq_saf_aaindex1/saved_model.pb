??4
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??1
~
aux_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *"
shared_nameaux_output/kernel
w
%aux_output/kernel/Read/ReadVariableOpReadVariableOpaux_output/kernel*
_output_shapes

: *
dtype0
v
aux_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameaux_output/bias
o
#aux_output/bias/Read/ReadVariableOpReadVariableOpaux_output/bias*
_output_shapes
:*
dtype0
|
dense_141/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namedense_141/kernel
u
$dense_141/kernel/Read/ReadVariableOpReadVariableOpdense_141/kernel*
_output_shapes

:@*
dtype0
t
dense_141/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_141/bias
m
"dense_141/bias/Read/ReadVariableOpReadVariableOpdense_141/bias*
_output_shapes
:@*
dtype0
|
dense_142/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_142/kernel
u
$dense_142/kernel/Read/ReadVariableOpReadVariableOpdense_142/kernel*
_output_shapes

:@@*
dtype0
t
dense_142/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_142/bias
m
"dense_142/bias/Read/ReadVariableOpReadVariableOpdense_142/bias*
_output_shapes
:@*
dtype0
|
dense_143/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*!
shared_namedense_143/kernel
u
$dense_143/kernel/Read/ReadVariableOpReadVariableOpdense_143/kernel*
_output_shapes

:@@*
dtype0
t
dense_143/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_143/bias
m
"dense_143/bias/Read/ReadVariableOpReadVariableOpdense_143/bias*
_output_shapes
:@*
dtype0
?
main_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*#
shared_namemain_output/kernel
y
&main_output/kernel/Read/ReadVariableOpReadVariableOpmain_output/kernel*
_output_shapes

:@*
dtype0
x
main_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namemain_output/bias
q
$main_output/bias/Read/ReadVariableOpReadVariableOpmain_output/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
7token_and_position_embedding_15/embedding_30/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *H
shared_name97token_and_position_embedding_15/embedding_30/embeddings
?
Ktoken_and_position_embedding_15/embedding_30/embeddings/Read/ReadVariableOpReadVariableOp7token_and_position_embedding_15/embedding_30/embeddings*
_output_shapes

: *
dtype0
?
7token_and_position_embedding_15/embedding_31/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *H
shared_name97token_and_position_embedding_15/embedding_31/embeddings
?
Ktoken_and_position_embedding_15/embedding_31/embeddings/Read/ReadVariableOpReadVariableOp7token_and_position_embedding_15/embedding_31/embeddings*
_output_shapes

:( *
dtype0
?
Btransformer_block_15/multi_head_self_attention_15/dense_135/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *S
shared_nameDBtransformer_block_15/multi_head_self_attention_15/dense_135/kernel
?
Vtransformer_block_15/multi_head_self_attention_15/dense_135/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_15/multi_head_self_attention_15/dense_135/kernel*
_output_shapes

:  *
dtype0
?
@transformer_block_15/multi_head_self_attention_15/dense_135/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_15/multi_head_self_attention_15/dense_135/bias
?
Ttransformer_block_15/multi_head_self_attention_15/dense_135/bias/Read/ReadVariableOpReadVariableOp@transformer_block_15/multi_head_self_attention_15/dense_135/bias*
_output_shapes
: *
dtype0
?
Btransformer_block_15/multi_head_self_attention_15/dense_136/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *S
shared_nameDBtransformer_block_15/multi_head_self_attention_15/dense_136/kernel
?
Vtransformer_block_15/multi_head_self_attention_15/dense_136/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_15/multi_head_self_attention_15/dense_136/kernel*
_output_shapes

:  *
dtype0
?
@transformer_block_15/multi_head_self_attention_15/dense_136/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_15/multi_head_self_attention_15/dense_136/bias
?
Ttransformer_block_15/multi_head_self_attention_15/dense_136/bias/Read/ReadVariableOpReadVariableOp@transformer_block_15/multi_head_self_attention_15/dense_136/bias*
_output_shapes
: *
dtype0
?
Btransformer_block_15/multi_head_self_attention_15/dense_137/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *S
shared_nameDBtransformer_block_15/multi_head_self_attention_15/dense_137/kernel
?
Vtransformer_block_15/multi_head_self_attention_15/dense_137/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_15/multi_head_self_attention_15/dense_137/kernel*
_output_shapes

:  *
dtype0
?
@transformer_block_15/multi_head_self_attention_15/dense_137/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_15/multi_head_self_attention_15/dense_137/bias
?
Ttransformer_block_15/multi_head_self_attention_15/dense_137/bias/Read/ReadVariableOpReadVariableOp@transformer_block_15/multi_head_self_attention_15/dense_137/bias*
_output_shapes
: *
dtype0
?
Btransformer_block_15/multi_head_self_attention_15/dense_138/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *S
shared_nameDBtransformer_block_15/multi_head_self_attention_15/dense_138/kernel
?
Vtransformer_block_15/multi_head_self_attention_15/dense_138/kernel/Read/ReadVariableOpReadVariableOpBtransformer_block_15/multi_head_self_attention_15/dense_138/kernel*
_output_shapes

:  *
dtype0
?
@transformer_block_15/multi_head_self_attention_15/dense_138/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@transformer_block_15/multi_head_self_attention_15/dense_138/bias
?
Ttransformer_block_15/multi_head_self_attention_15/dense_138/bias/Read/ReadVariableOpReadVariableOp@transformer_block_15/multi_head_self_attention_15/dense_138/bias*
_output_shapes
: *
dtype0
|
dense_139/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_139/kernel
u
$dense_139/kernel/Read/ReadVariableOpReadVariableOpdense_139/kernel*
_output_shapes

:  *
dtype0
t
dense_139/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_139/bias
m
"dense_139/bias/Read/ReadVariableOpReadVariableOpdense_139/bias*
_output_shapes
: *
dtype0
|
dense_140/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *!
shared_namedense_140/kernel
u
$dense_140/kernel/Read/ReadVariableOpReadVariableOpdense_140/kernel*
_output_shapes

:  *
dtype0
t
dense_140/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_140/bias
m
"dense_140/bias/Read/ReadVariableOpReadVariableOpdense_140/bias*
_output_shapes
: *
dtype0
?
1transformer_block_15/layer_normalization_30/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31transformer_block_15/layer_normalization_30/gamma
?
Etransformer_block_15/layer_normalization_30/gamma/Read/ReadVariableOpReadVariableOp1transformer_block_15/layer_normalization_30/gamma*
_output_shapes
: *
dtype0
?
0transformer_block_15/layer_normalization_30/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_15/layer_normalization_30/beta
?
Dtransformer_block_15/layer_normalization_30/beta/Read/ReadVariableOpReadVariableOp0transformer_block_15/layer_normalization_30/beta*
_output_shapes
: *
dtype0
?
1transformer_block_15/layer_normalization_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31transformer_block_15/layer_normalization_31/gamma
?
Etransformer_block_15/layer_normalization_31/gamma/Read/ReadVariableOpReadVariableOp1transformer_block_15/layer_normalization_31/gamma*
_output_shapes
: *
dtype0
?
0transformer_block_15/layer_normalization_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_15/layer_normalization_31/beta
?
Dtransformer_block_15/layer_normalization_31/beta/Read/ReadVariableOpReadVariableOp0transformer_block_15/layer_normalization_31/beta*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
?
Adam/aux_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/aux_output/kernel/m
?
,Adam/aux_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/aux_output/kernel/m*
_output_shapes

: *
dtype0
?
Adam/aux_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/aux_output/bias/m
}
*Adam/aux_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/aux_output/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_141/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_141/kernel/m
?
+Adam/dense_141/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_141/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_141/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_141/bias/m
{
)Adam/dense_141/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_141/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_142/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_142/kernel/m
?
+Adam/dense_142/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_142/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_142/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_142/bias/m
{
)Adam/dense_142/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_142/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_143/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_143/kernel/m
?
+Adam/dense_143/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_143/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_143/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_143/bias/m
{
)Adam/dense_143/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_143/bias/m*
_output_shapes
:@*
dtype0
?
Adam/main_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@**
shared_nameAdam/main_output/kernel/m
?
-Adam/main_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/main_output/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/main_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/main_output/bias/m

+Adam/main_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/main_output/bias/m*
_output_shapes
:*
dtype0
?
>Adam/token_and_position_embedding_15/embedding_30/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>Adam/token_and_position_embedding_15/embedding_30/embeddings/m
?
RAdam/token_and_position_embedding_15/embedding_30/embeddings/m/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_15/embedding_30/embeddings/m*
_output_shapes

: *
dtype0
?
>Adam/token_and_position_embedding_15/embedding_31/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *O
shared_name@>Adam/token_and_position_embedding_15/embedding_31/embeddings/m
?
RAdam/token_and_position_embedding_15/embedding_31/embeddings/m/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_15/embedding_31/embeddings/m*
_output_shapes

:( *
dtype0
?
IAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *Z
shared_nameKIAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/m
?
]Adam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/m*
_output_shapes

:  *
dtype0
?
GAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/m
?
[Adam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/m*
_output_shapes
: *
dtype0
?
IAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *Z
shared_nameKIAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/m
?
]Adam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/m*
_output_shapes

:  *
dtype0
?
GAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/m
?
[Adam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/m*
_output_shapes
: *
dtype0
?
IAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *Z
shared_nameKIAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/m
?
]Adam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/m*
_output_shapes

:  *
dtype0
?
GAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/m
?
[Adam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/m*
_output_shapes
: *
dtype0
?
IAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *Z
shared_nameKIAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/m
?
]Adam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/m/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/m*
_output_shapes

:  *
dtype0
?
GAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/m
?
[Adam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/m/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_139/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_139/kernel/m
?
+Adam/dense_139/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_139/kernel/m*
_output_shapes

:  *
dtype0
?
Adam/dense_139/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_139/bias/m
{
)Adam/dense_139/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_139/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_140/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_140/kernel/m
?
+Adam/dense_140/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_140/kernel/m*
_output_shapes

:  *
dtype0
?
Adam/dense_140/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_140/bias/m
{
)Adam/dense_140/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_140/bias/m*
_output_shapes
: *
dtype0
?
8Adam/transformer_block_15/layer_normalization_30/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/transformer_block_15/layer_normalization_30/gamma/m
?
LAdam/transformer_block_15/layer_normalization_30/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_block_15/layer_normalization_30/gamma/m*
_output_shapes
: *
dtype0
?
7Adam/transformer_block_15/layer_normalization_30/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_15/layer_normalization_30/beta/m
?
KAdam/transformer_block_15/layer_normalization_30/beta/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_15/layer_normalization_30/beta/m*
_output_shapes
: *
dtype0
?
8Adam/transformer_block_15/layer_normalization_31/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/transformer_block_15/layer_normalization_31/gamma/m
?
LAdam/transformer_block_15/layer_normalization_31/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/transformer_block_15/layer_normalization_31/gamma/m*
_output_shapes
: *
dtype0
?
7Adam/transformer_block_15/layer_normalization_31/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_15/layer_normalization_31/beta/m
?
KAdam/transformer_block_15/layer_normalization_31/beta/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_15/layer_normalization_31/beta/m*
_output_shapes
: *
dtype0
?
Adam/aux_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *)
shared_nameAdam/aux_output/kernel/v
?
,Adam/aux_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/aux_output/kernel/v*
_output_shapes

: *
dtype0
?
Adam/aux_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/aux_output/bias/v
}
*Adam/aux_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/aux_output/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_141/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_141/kernel/v
?
+Adam/dense_141/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_141/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_141/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_141/bias/v
{
)Adam/dense_141/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_141/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_142/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_142/kernel/v
?
+Adam/dense_142/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_142/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_142/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_142/bias/v
{
)Adam/dense_142/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_142/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_143/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*(
shared_nameAdam/dense_143/kernel/v
?
+Adam/dense_143/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_143/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_143/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_143/bias/v
{
)Adam/dense_143/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_143/bias/v*
_output_shapes
:@*
dtype0
?
Adam/main_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@**
shared_nameAdam/main_output/kernel/v
?
-Adam/main_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/main_output/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/main_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/main_output/bias/v

+Adam/main_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/main_output/bias/v*
_output_shapes
:*
dtype0
?
>Adam/token_and_position_embedding_15/embedding_30/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *O
shared_name@>Adam/token_and_position_embedding_15/embedding_30/embeddings/v
?
RAdam/token_and_position_embedding_15/embedding_30/embeddings/v/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_15/embedding_30/embeddings/v*
_output_shapes

: *
dtype0
?
>Adam/token_and_position_embedding_15/embedding_31/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *O
shared_name@>Adam/token_and_position_embedding_15/embedding_31/embeddings/v
?
RAdam/token_and_position_embedding_15/embedding_31/embeddings/v/Read/ReadVariableOpReadVariableOp>Adam/token_and_position_embedding_15/embedding_31/embeddings/v*
_output_shapes

:( *
dtype0
?
IAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *Z
shared_nameKIAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/v
?
]Adam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/v*
_output_shapes

:  *
dtype0
?
GAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/v
?
[Adam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/v*
_output_shapes
: *
dtype0
?
IAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *Z
shared_nameKIAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/v
?
]Adam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/v*
_output_shapes

:  *
dtype0
?
GAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/v
?
[Adam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/v*
_output_shapes
: *
dtype0
?
IAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *Z
shared_nameKIAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/v
?
]Adam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/v*
_output_shapes

:  *
dtype0
?
GAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/v
?
[Adam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/v*
_output_shapes
: *
dtype0
?
IAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *Z
shared_nameKIAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/v
?
]Adam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/v/Read/ReadVariableOpReadVariableOpIAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/v*
_output_shapes

:  *
dtype0
?
GAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *X
shared_nameIGAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/v
?
[Adam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/v/Read/ReadVariableOpReadVariableOpGAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_139/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_139/kernel/v
?
+Adam/dense_139/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_139/kernel/v*
_output_shapes

:  *
dtype0
?
Adam/dense_139/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_139/bias/v
{
)Adam/dense_139/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_139/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_140/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *(
shared_nameAdam/dense_140/kernel/v
?
+Adam/dense_140/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_140/kernel/v*
_output_shapes

:  *
dtype0
?
Adam/dense_140/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_140/bias/v
{
)Adam/dense_140/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_140/bias/v*
_output_shapes
: *
dtype0
?
8Adam/transformer_block_15/layer_normalization_30/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/transformer_block_15/layer_normalization_30/gamma/v
?
LAdam/transformer_block_15/layer_normalization_30/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_block_15/layer_normalization_30/gamma/v*
_output_shapes
: *
dtype0
?
7Adam/transformer_block_15/layer_normalization_30/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_15/layer_normalization_30/beta/v
?
KAdam/transformer_block_15/layer_normalization_30/beta/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_15/layer_normalization_30/beta/v*
_output_shapes
: *
dtype0
?
8Adam/transformer_block_15/layer_normalization_31/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/transformer_block_15/layer_normalization_31/gamma/v
?
LAdam/transformer_block_15/layer_normalization_31/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/transformer_block_15/layer_normalization_31/gamma/v*
_output_shapes
: *
dtype0
?
7Adam/transformer_block_15/layer_normalization_31/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_15/layer_normalization_31/beta/v
?
KAdam/transformer_block_15/layer_normalization_31/beta/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_15/layer_normalization_31/beta/v*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??Bܳ BԳ
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
n
	token_emb
pos_emb
regularization_losses
trainable_variables
	variables
	keras_api
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
regularization_losses
 trainable_variables
!	variables
"	keras_api
R
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
 
 
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
h

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
?
Iiter

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_rate'm?(m?1m?2m?7m?8m?=m?>m?Cm?Dm?Nm?Om?Pm?Qm?Rm?Sm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?]m?^m?_m?'v?(v?1v?2v?7v?8v?=v?>v?Cv?Dv?Nv?Ov?Pv?Qv?Rv?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?]v?^v?_v?
 
?
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
Z12
[13
\14
]15
^16
_17
'18
(19
120
221
722
823
=24
>25
C26
D27
?
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
Z12
[13
\14
]15
^16
_17
'18
(19
120
221
722
823
=24
>25
C26
D27
?
regularization_losses
trainable_variables
`layer_metrics
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
	variables
 
b
N
embeddings
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
b
O
embeddings
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
 

N0
O1

N0
O1
?
regularization_losses
trainable_variables
mlayer_metrics

nlayers
onon_trainable_variables
pmetrics
qlayer_regularization_losses
	variables
?
rquery_dense
s	key_dense
tvalue_dense
ucombine_heads
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
?
zlayer_with_weights-0
zlayer-0
{layer_with_weights-1
{layer-1
|regularization_losses
}trainable_variables
~	variables
	keras_api
v
	?axis
	\gamma
]beta
?regularization_losses
?trainable_variables
?	variables
?	keras_api
v
	?axis
	^gamma
_beta
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
V
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
v
P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
Z10
[11
\12
]13
^14
_15
v
P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
Z10
[11
\12
]13
^14
_15
?
regularization_losses
 trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
!	variables
 
 
 
?
#regularization_losses
$trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
%	variables
][
VARIABLE_VALUEaux_output/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEaux_output/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
?
)regularization_losses
*trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
+	variables
 
 
 
?
-regularization_losses
.trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
/	variables
\Z
VARIABLE_VALUEdense_141/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_141/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
?
3regularization_losses
4trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
5	variables
\Z
VARIABLE_VALUEdense_142/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_142/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
?
9regularization_losses
:trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
;	variables
\Z
VARIABLE_VALUEdense_143/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_143/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
?
?regularization_losses
@trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
A	variables
^\
VARIABLE_VALUEmain_output/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEmain_output/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
?
Eregularization_losses
Ftrainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
G	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE7token_and_position_embedding_15/embedding_30/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE7token_and_position_embedding_15/embedding_31/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBtransformer_block_15/multi_head_self_attention_15/dense_135/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@transformer_block_15/multi_head_self_attention_15/dense_135/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBtransformer_block_15/multi_head_self_attention_15/dense_136/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@transformer_block_15/multi_head_self_attention_15/dense_136/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBtransformer_block_15/multi_head_self_attention_15/dense_137/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@transformer_block_15/multi_head_self_attention_15/dense_137/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEBtransformer_block_15/multi_head_self_attention_15/dense_138/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE@transformer_block_15/multi_head_self_attention_15/dense_138/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_139/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_139/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_140/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_140/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE1transformer_block_15/layer_normalization_30/gamma1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_15/layer_normalization_30/beta1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE1transformer_block_15/layer_normalization_31/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_15/layer_normalization_31/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
(
?0
?1
?2
?3
?4
 
 

N0

N0
?
eregularization_losses
ftrainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
g	variables
 

O0

O0
?
iregularization_losses
jtrainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
k	variables
 

0
1
 
 
 
l

Pkernel
Qbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Rkernel
Sbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Tkernel
Ubias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Vkernel
Wbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
8
P0
Q1
R2
S3
T4
U5
V6
W7
8
P0
Q1
R2
S3
T4
U5
V6
W7
?
vregularization_losses
wtrainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
x	variables
l

Xkernel
Ybias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Zkernel
[bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 

X0
Y1
Z2
[3

X0
Y1
Z2
[3
?
|regularization_losses
}trainable_variables
?layer_metrics
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
~	variables
 
 

\0
]1

\0
]1
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 
 

^0
_1

^0
_1
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 
 
 
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 
 
 
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 

P0
Q1

P0
Q1
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 

R0
S1

R0
S1
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 

T0
U1

T0
U1
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 

V0
W1

V0
W1
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 

r0
s1
t2
u3
 
 
 
 

X0
Y1

X0
Y1
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 

Z0
[1

Z0
[1
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
 
 

z0
{1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?~
VARIABLE_VALUEAdam/aux_output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/aux_output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_141/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_141/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_142/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_142/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_143/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_143/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/main_output/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/main_output/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_15/embedding_30/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_15/embedding_31/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_139/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_139/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_140/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_140/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/transformer_block_15/layer_normalization_30/gamma/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_15/layer_normalization_30/beta/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/transformer_block_15/layer_normalization_31/gamma/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_15/layer_normalization_31/beta/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/aux_output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/aux_output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_141/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_141/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_142/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_142/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_143/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_143/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/main_output/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/main_output/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_15/embedding_30/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE>Adam/token_and_position_embedding_15/embedding_31/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEIAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEGAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_139/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_139/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_140/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_140/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/transformer_block_15/layer_normalization_30/gamma/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_15/layer_normalization_30/beta/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/transformer_block_15/layer_normalization_31/gamma/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_15/layer_normalization_31/beta/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_aaindex_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_aux_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_16Placeholder*'
_output_shapes
:?????????(*
dtype0*
shape:?????????(
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_aaindex_inputserving_default_aux_inputserving_default_input_167token_and_position_embedding_15/embedding_31/embeddings7token_and_position_embedding_15/embedding_30/embeddingsBtransformer_block_15/multi_head_self_attention_15/dense_135/kernel@transformer_block_15/multi_head_self_attention_15/dense_135/biasBtransformer_block_15/multi_head_self_attention_15/dense_136/kernel@transformer_block_15/multi_head_self_attention_15/dense_136/biasBtransformer_block_15/multi_head_self_attention_15/dense_137/kernel@transformer_block_15/multi_head_self_attention_15/dense_137/biasBtransformer_block_15/multi_head_self_attention_15/dense_138/kernel@transformer_block_15/multi_head_self_attention_15/dense_138/bias1transformer_block_15/layer_normalization_30/gamma0transformer_block_15/layer_normalization_30/betadense_139/kerneldense_139/biasdense_140/kerneldense_140/bias1transformer_block_15/layer_normalization_31/gamma0transformer_block_15/layer_normalization_31/betaaux_output/kernelaux_output/biasdense_141/kerneldense_141/biasdense_142/kerneldense_142/biasdense_143/kerneldense_143/biasmain_output/kernelmain_output/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_6976554
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?0
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%aux_output/kernel/Read/ReadVariableOp#aux_output/bias/Read/ReadVariableOp$dense_141/kernel/Read/ReadVariableOp"dense_141/bias/Read/ReadVariableOp$dense_142/kernel/Read/ReadVariableOp"dense_142/bias/Read/ReadVariableOp$dense_143/kernel/Read/ReadVariableOp"dense_143/bias/Read/ReadVariableOp&main_output/kernel/Read/ReadVariableOp$main_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpKtoken_and_position_embedding_15/embedding_30/embeddings/Read/ReadVariableOpKtoken_and_position_embedding_15/embedding_31/embeddings/Read/ReadVariableOpVtransformer_block_15/multi_head_self_attention_15/dense_135/kernel/Read/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_135/bias/Read/ReadVariableOpVtransformer_block_15/multi_head_self_attention_15/dense_136/kernel/Read/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_136/bias/Read/ReadVariableOpVtransformer_block_15/multi_head_self_attention_15/dense_137/kernel/Read/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_137/bias/Read/ReadVariableOpVtransformer_block_15/multi_head_self_attention_15/dense_138/kernel/Read/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_138/bias/Read/ReadVariableOp$dense_139/kernel/Read/ReadVariableOp"dense_139/bias/Read/ReadVariableOp$dense_140/kernel/Read/ReadVariableOp"dense_140/bias/Read/ReadVariableOpEtransformer_block_15/layer_normalization_30/gamma/Read/ReadVariableOpDtransformer_block_15/layer_normalization_30/beta/Read/ReadVariableOpEtransformer_block_15/layer_normalization_31/gamma/Read/ReadVariableOpDtransformer_block_15/layer_normalization_31/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOp,Adam/aux_output/kernel/m/Read/ReadVariableOp*Adam/aux_output/bias/m/Read/ReadVariableOp+Adam/dense_141/kernel/m/Read/ReadVariableOp)Adam/dense_141/bias/m/Read/ReadVariableOp+Adam/dense_142/kernel/m/Read/ReadVariableOp)Adam/dense_142/bias/m/Read/ReadVariableOp+Adam/dense_143/kernel/m/Read/ReadVariableOp)Adam/dense_143/bias/m/Read/ReadVariableOp-Adam/main_output/kernel/m/Read/ReadVariableOp+Adam/main_output/bias/m/Read/ReadVariableOpRAdam/token_and_position_embedding_15/embedding_30/embeddings/m/Read/ReadVariableOpRAdam/token_and_position_embedding_15/embedding_31/embeddings/m/Read/ReadVariableOp]Adam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/m/Read/ReadVariableOp[Adam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/m/Read/ReadVariableOp]Adam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/m/Read/ReadVariableOp[Adam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/m/Read/ReadVariableOp]Adam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/m/Read/ReadVariableOp[Adam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/m/Read/ReadVariableOp]Adam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/m/Read/ReadVariableOp[Adam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/m/Read/ReadVariableOp+Adam/dense_139/kernel/m/Read/ReadVariableOp)Adam/dense_139/bias/m/Read/ReadVariableOp+Adam/dense_140/kernel/m/Read/ReadVariableOp)Adam/dense_140/bias/m/Read/ReadVariableOpLAdam/transformer_block_15/layer_normalization_30/gamma/m/Read/ReadVariableOpKAdam/transformer_block_15/layer_normalization_30/beta/m/Read/ReadVariableOpLAdam/transformer_block_15/layer_normalization_31/gamma/m/Read/ReadVariableOpKAdam/transformer_block_15/layer_normalization_31/beta/m/Read/ReadVariableOp,Adam/aux_output/kernel/v/Read/ReadVariableOp*Adam/aux_output/bias/v/Read/ReadVariableOp+Adam/dense_141/kernel/v/Read/ReadVariableOp)Adam/dense_141/bias/v/Read/ReadVariableOp+Adam/dense_142/kernel/v/Read/ReadVariableOp)Adam/dense_142/bias/v/Read/ReadVariableOp+Adam/dense_143/kernel/v/Read/ReadVariableOp)Adam/dense_143/bias/v/Read/ReadVariableOp-Adam/main_output/kernel/v/Read/ReadVariableOp+Adam/main_output/bias/v/Read/ReadVariableOpRAdam/token_and_position_embedding_15/embedding_30/embeddings/v/Read/ReadVariableOpRAdam/token_and_position_embedding_15/embedding_31/embeddings/v/Read/ReadVariableOp]Adam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/v/Read/ReadVariableOp[Adam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/v/Read/ReadVariableOp]Adam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/v/Read/ReadVariableOp[Adam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/v/Read/ReadVariableOp]Adam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/v/Read/ReadVariableOp[Adam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/v/Read/ReadVariableOp]Adam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/v/Read/ReadVariableOp[Adam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/v/Read/ReadVariableOp+Adam/dense_139/kernel/v/Read/ReadVariableOp)Adam/dense_139/bias/v/Read/ReadVariableOp+Adam/dense_140/kernel/v/Read/ReadVariableOp)Adam/dense_140/bias/v/Read/ReadVariableOpLAdam/transformer_block_15/layer_normalization_30/gamma/v/Read/ReadVariableOpKAdam/transformer_block_15/layer_normalization_30/beta/v/Read/ReadVariableOpLAdam/transformer_block_15/layer_normalization_31/gamma/v/Read/ReadVariableOpKAdam/transformer_block_15/layer_normalization_31/beta/v/Read/ReadVariableOpConst*p
Tini
g2e	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_6978598
? 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameaux_output/kernelaux_output/biasdense_141/kerneldense_141/biasdense_142/kerneldense_142/biasdense_143/kerneldense_143/biasmain_output/kernelmain_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate7token_and_position_embedding_15/embedding_30/embeddings7token_and_position_embedding_15/embedding_31/embeddingsBtransformer_block_15/multi_head_self_attention_15/dense_135/kernel@transformer_block_15/multi_head_self_attention_15/dense_135/biasBtransformer_block_15/multi_head_self_attention_15/dense_136/kernel@transformer_block_15/multi_head_self_attention_15/dense_136/biasBtransformer_block_15/multi_head_self_attention_15/dense_137/kernel@transformer_block_15/multi_head_self_attention_15/dense_137/biasBtransformer_block_15/multi_head_self_attention_15/dense_138/kernel@transformer_block_15/multi_head_self_attention_15/dense_138/biasdense_139/kerneldense_139/biasdense_140/kerneldense_140/bias1transformer_block_15/layer_normalization_30/gamma0transformer_block_15/layer_normalization_30/beta1transformer_block_15/layer_normalization_31/gamma0transformer_block_15/layer_normalization_31/betatotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4Adam/aux_output/kernel/mAdam/aux_output/bias/mAdam/dense_141/kernel/mAdam/dense_141/bias/mAdam/dense_142/kernel/mAdam/dense_142/bias/mAdam/dense_143/kernel/mAdam/dense_143/bias/mAdam/main_output/kernel/mAdam/main_output/bias/m>Adam/token_and_position_embedding_15/embedding_30/embeddings/m>Adam/token_and_position_embedding_15/embedding_31/embeddings/mIAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/mGAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/mIAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/mGAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/mIAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/mGAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/mIAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/mGAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/mAdam/dense_139/kernel/mAdam/dense_139/bias/mAdam/dense_140/kernel/mAdam/dense_140/bias/m8Adam/transformer_block_15/layer_normalization_30/gamma/m7Adam/transformer_block_15/layer_normalization_30/beta/m8Adam/transformer_block_15/layer_normalization_31/gamma/m7Adam/transformer_block_15/layer_normalization_31/beta/mAdam/aux_output/kernel/vAdam/aux_output/bias/vAdam/dense_141/kernel/vAdam/dense_141/bias/vAdam/dense_142/kernel/vAdam/dense_142/bias/vAdam/dense_143/kernel/vAdam/dense_143/bias/vAdam/main_output/kernel/vAdam/main_output/bias/v>Adam/token_and_position_embedding_15/embedding_30/embeddings/v>Adam/token_and_position_embedding_15/embedding_31/embeddings/vIAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/vGAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/vIAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/vGAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/vIAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/vGAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/vIAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/vGAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/vAdam/dense_139/kernel/vAdam/dense_139/bias/vAdam/dense_140/kernel/vAdam/dense_140/bias/v8Adam/transformer_block_15/layer_normalization_30/gamma/v7Adam/transformer_block_15/layer_normalization_30/beta/v8Adam/transformer_block_15/layer_normalization_31/gamma/v7Adam/transformer_block_15/layer_normalization_31/beta/v*o
Tinh
f2d*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_6978905??-
?
?
6__inference_transformer_block_15_layer_call_fn_6977882

inputs
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:  

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_69754652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?>
?
E__inference_model_15_layer_call_and_return_conditional_losses_6976211

inputs
inputs_1
inputs_29
'token_and_position_embedding_15_6976144:( 9
'token_and_position_embedding_15_6976146: .
transformer_block_15_6976149:  *
transformer_block_15_6976151: .
transformer_block_15_6976153:  *
transformer_block_15_6976155: .
transformer_block_15_6976157:  *
transformer_block_15_6976159: .
transformer_block_15_6976161:  *
transformer_block_15_6976163: *
transformer_block_15_6976165: *
transformer_block_15_6976167: .
transformer_block_15_6976169:  *
transformer_block_15_6976171: .
transformer_block_15_6976173:  *
transformer_block_15_6976175: *
transformer_block_15_6976177: *
transformer_block_15_6976179: $
aux_output_6976183:  
aux_output_6976185:#
dense_141_6976189:@
dense_141_6976191:@#
dense_142_6976194:@@
dense_142_6976196:@#
dense_143_6976199:@@
dense_143_6976201:@%
main_output_6976204:@!
main_output_6976206:
identity

identity_1??"aux_output/StatefulPartitionedCall?!dense_141/StatefulPartitionedCall?!dense_142/StatefulPartitionedCall?!dense_143/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?7token_and_position_embedding_15/StatefulPartitionedCall?,transformer_block_15/StatefulPartitionedCall?
7token_and_position_embedding_15/StatefulPartitionedCallStatefulPartitionedCallinputs'token_and_position_embedding_15_6976144'token_and_position_embedding_15_6976146*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_token_and_position_embedding_15_layer_call_and_return_conditional_losses_697521529
7token_and_position_embedding_15/StatefulPartitionedCall?
,transformer_block_15/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_15/StatefulPartitionedCall:output:0transformer_block_15_6976149transformer_block_15_6976151transformer_block_15_6976153transformer_block_15_6976155transformer_block_15_6976157transformer_block_15_6976159transformer_block_15_6976161transformer_block_15_6976163transformer_block_15_6976165transformer_block_15_6976167transformer_block_15_6976169transformer_block_15_6976171transformer_block_15_6976173transformer_block_15_6976175transformer_block_15_6976177transformer_block_15_6976179*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_69760252.
,transformer_block_15/StatefulPartitionedCall?
+global_average_pooling1d_15/PartitionedCallPartitionedCall5transformer_block_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *a
f\RZ
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_69755042-
+global_average_pooling1d_15/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_15/PartitionedCall:output:0aux_output_6976183aux_output_6976185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_aux_output_layer_call_and_return_conditional_losses_69755172$
"aux_output/StatefulPartitionedCall?
concatenate_15/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_15_layer_call_and_return_conditional_losses_69755312 
concatenate_15/PartitionedCall?
!dense_141/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0dense_141_6976189dense_141_6976191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_141_layer_call_and_return_conditional_losses_69755442#
!dense_141/StatefulPartitionedCall?
!dense_142/StatefulPartitionedCallStatefulPartitionedCall*dense_141/StatefulPartitionedCall:output:0dense_142_6976194dense_142_6976196*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_142_layer_call_and_return_conditional_losses_69755612#
!dense_142/StatefulPartitionedCall?
!dense_143/StatefulPartitionedCallStatefulPartitionedCall*dense_142/StatefulPartitionedCall:output:0dense_143_6976199dense_143_6976201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_143_layer_call_and_return_conditional_losses_69755782#
!dense_143/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall*dense_143/StatefulPartitionedCall:output:0main_output_6976204main_output_6976206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_main_output_layer_call_and_return_conditional_losses_69755952%
#main_output/StatefulPartitionedCall?
IdentityIdentity,main_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity+aux_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp#^aux_output/StatefulPartitionedCall"^dense_141/StatefulPartitionedCall"^dense_142/StatefulPartitionedCall"^dense_143/StatefulPartitionedCall$^main_output/StatefulPartitionedCall8^token_and_position_embedding_15/StatefulPartitionedCall-^transformer_block_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2F
!dense_141/StatefulPartitionedCall!dense_141/StatefulPartitionedCall2F
!dense_142/StatefulPartitionedCall!dense_142/StatefulPartitionedCall2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2r
7token_and_position_embedding_15/StatefulPartitionedCall7token_and_position_embedding_15/StatefulPartitionedCall2\
,transformer_block_15/StatefulPartitionedCall,transformer_block_15/StatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Y
=__inference_global_average_pooling1d_15_layer_call_fn_6977936

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *a
f\RZ
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_69751662
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_model_15_layer_call_fn_6975664
input_16
	aux_input
aaindex_input
unknown:( 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18:

unknown_19:@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_16	aux_inputaaindex_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_15_layer_call_and_return_conditional_losses_69756032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
input_16:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input:VR
'
_output_shapes
:?????????
'
_user_specified_nameaaindex_input
?
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_6975156
dense_139_input#
dense_139_6975145:  
dense_139_6975147: #
dense_140_6975150:  
dense_140_6975152: 
identity??!dense_139/StatefulPartitionedCall?!dense_140/StatefulPartitionedCall?
!dense_139/StatefulPartitionedCallStatefulPartitionedCalldense_139_inputdense_139_6975145dense_139_6975147*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_139_layer_call_and_return_conditional_losses_69750012#
!dense_139/StatefulPartitionedCall?
!dense_140/StatefulPartitionedCallStatefulPartitionedCall*dense_139/StatefulPartitionedCall:output:0dense_140_6975150dense_140_6975152*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_140_layer_call_and_return_conditional_losses_69750372#
!dense_140/StatefulPartitionedCall?
IdentityIdentity*dense_140/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp"^dense_139/StatefulPartitionedCall"^dense_140/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2F
!dense_139/StatefulPartitionedCall!dense_139/StatefulPartitionedCall2F
!dense_140/StatefulPartitionedCall!dense_140/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????( 
)
_user_specified_namedense_139_input
?
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_6975104

inputs#
dense_139_6975093:  
dense_139_6975095: #
dense_140_6975098:  
dense_140_6975100: 
identity??!dense_139/StatefulPartitionedCall?!dense_140/StatefulPartitionedCall?
!dense_139/StatefulPartitionedCallStatefulPartitionedCallinputsdense_139_6975093dense_139_6975095*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_139_layer_call_and_return_conditional_losses_69750012#
!dense_139/StatefulPartitionedCall?
!dense_140/StatefulPartitionedCallStatefulPartitionedCall*dense_139/StatefulPartitionedCall:output:0dense_140_6975098dense_140_6975100*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_140_layer_call_and_return_conditional_losses_69750372#
!dense_140/StatefulPartitionedCall?
IdentityIdentity*dense_140/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp"^dense_139/StatefulPartitionedCall"^dense_140/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2F
!dense_139/StatefulPartitionedCall!dense_139/StatefulPartitionedCall2F
!dense_140/StatefulPartitionedCall!dense_140/StatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
F__inference_dense_141_layer_call_and_return_conditional_losses_6975544

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_main_output_layer_call_fn_6978056

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_main_output_layer_call_and_return_conditional_losses_69755952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
+__inference_dense_140_layer_call_fn_6978275

inputs
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_140_layer_call_and_return_conditional_losses_69750372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????( : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
K__inference_concatenate_15_layer_call_and_return_conditional_losses_6977969
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?
?
6__inference_transformer_block_15_layer_call_fn_6977919

inputs
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:  

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13: 

unknown_14: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_69760252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_6975142
dense_139_input#
dense_139_6975131:  
dense_139_6975133: #
dense_140_6975136:  
dense_140_6975138: 
identity??!dense_139/StatefulPartitionedCall?!dense_140/StatefulPartitionedCall?
!dense_139/StatefulPartitionedCallStatefulPartitionedCalldense_139_inputdense_139_6975131dense_139_6975133*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_139_layer_call_and_return_conditional_losses_69750012#
!dense_139/StatefulPartitionedCall?
!dense_140/StatefulPartitionedCallStatefulPartitionedCall*dense_139/StatefulPartitionedCall:output:0dense_140_6975136dense_140_6975138*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_140_layer_call_and_return_conditional_losses_69750372#
!dense_140/StatefulPartitionedCall?
IdentityIdentity*dense_140/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp"^dense_139/StatefulPartitionedCall"^dense_140/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2F
!dense_139/StatefulPartitionedCall!dense_139/StatefulPartitionedCall2F
!dense_140/StatefulPartitionedCall!dense_140/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????( 
)
_user_specified_namedense_139_input
?
?
G__inference_aux_output_layer_call_and_return_conditional_losses_6975517

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?>
?
E__inference_model_15_layer_call_and_return_conditional_losses_6976481
input_16
	aux_input
aaindex_input9
'token_and_position_embedding_15_6976414:( 9
'token_and_position_embedding_15_6976416: .
transformer_block_15_6976419:  *
transformer_block_15_6976421: .
transformer_block_15_6976423:  *
transformer_block_15_6976425: .
transformer_block_15_6976427:  *
transformer_block_15_6976429: .
transformer_block_15_6976431:  *
transformer_block_15_6976433: *
transformer_block_15_6976435: *
transformer_block_15_6976437: .
transformer_block_15_6976439:  *
transformer_block_15_6976441: .
transformer_block_15_6976443:  *
transformer_block_15_6976445: *
transformer_block_15_6976447: *
transformer_block_15_6976449: $
aux_output_6976453:  
aux_output_6976455:#
dense_141_6976459:@
dense_141_6976461:@#
dense_142_6976464:@@
dense_142_6976466:@#
dense_143_6976469:@@
dense_143_6976471:@%
main_output_6976474:@!
main_output_6976476:
identity

identity_1??"aux_output/StatefulPartitionedCall?!dense_141/StatefulPartitionedCall?!dense_142/StatefulPartitionedCall?!dense_143/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?7token_and_position_embedding_15/StatefulPartitionedCall?,transformer_block_15/StatefulPartitionedCall?
7token_and_position_embedding_15/StatefulPartitionedCallStatefulPartitionedCallinput_16'token_and_position_embedding_15_6976414'token_and_position_embedding_15_6976416*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_token_and_position_embedding_15_layer_call_and_return_conditional_losses_697521529
7token_and_position_embedding_15/StatefulPartitionedCall?
,transformer_block_15/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_15/StatefulPartitionedCall:output:0transformer_block_15_6976419transformer_block_15_6976421transformer_block_15_6976423transformer_block_15_6976425transformer_block_15_6976427transformer_block_15_6976429transformer_block_15_6976431transformer_block_15_6976433transformer_block_15_6976435transformer_block_15_6976437transformer_block_15_6976439transformer_block_15_6976441transformer_block_15_6976443transformer_block_15_6976445transformer_block_15_6976447transformer_block_15_6976449*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_69760252.
,transformer_block_15/StatefulPartitionedCall?
+global_average_pooling1d_15/PartitionedCallPartitionedCall5transformer_block_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *a
f\RZ
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_69755042-
+global_average_pooling1d_15/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_15/PartitionedCall:output:0aux_output_6976453aux_output_6976455*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_aux_output_layer_call_and_return_conditional_losses_69755172$
"aux_output/StatefulPartitionedCall?
concatenate_15/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0	aux_inputaaindex_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_15_layer_call_and_return_conditional_losses_69755312 
concatenate_15/PartitionedCall?
!dense_141/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0dense_141_6976459dense_141_6976461*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_141_layer_call_and_return_conditional_losses_69755442#
!dense_141/StatefulPartitionedCall?
!dense_142/StatefulPartitionedCallStatefulPartitionedCall*dense_141/StatefulPartitionedCall:output:0dense_142_6976464dense_142_6976466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_142_layer_call_and_return_conditional_losses_69755612#
!dense_142/StatefulPartitionedCall?
!dense_143/StatefulPartitionedCallStatefulPartitionedCall*dense_142/StatefulPartitionedCall:output:0dense_143_6976469dense_143_6976471*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_143_layer_call_and_return_conditional_losses_69755782#
!dense_143/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall*dense_143/StatefulPartitionedCall:output:0main_output_6976474main_output_6976476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_main_output_layer_call_and_return_conditional_losses_69755952%
#main_output/StatefulPartitionedCall?
IdentityIdentity,main_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity+aux_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp#^aux_output/StatefulPartitionedCall"^dense_141/StatefulPartitionedCall"^dense_142/StatefulPartitionedCall"^dense_143/StatefulPartitionedCall$^main_output/StatefulPartitionedCall8^token_and_position_embedding_15/StatefulPartitionedCall-^transformer_block_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2F
!dense_141/StatefulPartitionedCall!dense_141/StatefulPartitionedCall2F
!dense_142/StatefulPartitionedCall!dense_142/StatefulPartitionedCall2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2r
7token_and_position_embedding_15/StatefulPartitionedCall7token_and_position_embedding_15/StatefulPartitionedCall2\
,transformer_block_15/StatefulPartitionedCall,transformer_block_15/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
input_16:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input:VR
'
_output_shapes
:?????????
'
_user_specified_nameaaindex_input
?
?
H__inference_main_output_layer_call_and_return_conditional_losses_6978047

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_model_15_layer_call_fn_6977310
inputs_0
inputs_1
inputs_2
unknown:( 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18:

unknown_19:@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_15_layer_call_and_return_conditional_losses_69762112
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?
?
+__inference_dense_139_layer_call_fn_6978236

inputs
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_139_layer_call_and_return_conditional_losses_69750012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????( : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?!
?
F__inference_dense_139_layer_call_and_return_conditional_losses_6978227

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????( : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
??
?!
E__inference_model_15_layer_call_and_return_conditional_losses_6976860
inputs_0
inputs_1
inputs_2W
Etoken_and_position_embedding_15_embedding_31_embedding_lookup_6976567:( W
Etoken_and_position_embedding_15_embedding_30_embedding_lookup_6976573: o
]transformer_block_15_multi_head_self_attention_15_dense_135_tensordot_readvariableop_resource:  i
[transformer_block_15_multi_head_self_attention_15_dense_135_biasadd_readvariableop_resource: o
]transformer_block_15_multi_head_self_attention_15_dense_136_tensordot_readvariableop_resource:  i
[transformer_block_15_multi_head_self_attention_15_dense_136_biasadd_readvariableop_resource: o
]transformer_block_15_multi_head_self_attention_15_dense_137_tensordot_readvariableop_resource:  i
[transformer_block_15_multi_head_self_attention_15_dense_137_biasadd_readvariableop_resource: o
]transformer_block_15_multi_head_self_attention_15_dense_138_tensordot_readvariableop_resource:  i
[transformer_block_15_multi_head_self_attention_15_dense_138_biasadd_readvariableop_resource: _
Qtransformer_block_15_layer_normalization_30_batchnorm_mul_readvariableop_resource: [
Mtransformer_block_15_layer_normalization_30_batchnorm_readvariableop_resource: `
Ntransformer_block_15_sequential_15_dense_139_tensordot_readvariableop_resource:  Z
Ltransformer_block_15_sequential_15_dense_139_biasadd_readvariableop_resource: `
Ntransformer_block_15_sequential_15_dense_140_tensordot_readvariableop_resource:  Z
Ltransformer_block_15_sequential_15_dense_140_biasadd_readvariableop_resource: _
Qtransformer_block_15_layer_normalization_31_batchnorm_mul_readvariableop_resource: [
Mtransformer_block_15_layer_normalization_31_batchnorm_readvariableop_resource: ;
)aux_output_matmul_readvariableop_resource: 8
*aux_output_biasadd_readvariableop_resource::
(dense_141_matmul_readvariableop_resource:@7
)dense_141_biasadd_readvariableop_resource:@:
(dense_142_matmul_readvariableop_resource:@@7
)dense_142_biasadd_readvariableop_resource:@:
(dense_143_matmul_readvariableop_resource:@@7
)dense_143_biasadd_readvariableop_resource:@<
*main_output_matmul_readvariableop_resource:@9
+main_output_biasadd_readvariableop_resource:
identity

identity_1??!aux_output/BiasAdd/ReadVariableOp? aux_output/MatMul/ReadVariableOp? dense_141/BiasAdd/ReadVariableOp?dense_141/MatMul/ReadVariableOp? dense_142/BiasAdd/ReadVariableOp?dense_142/MatMul/ReadVariableOp? dense_143/BiasAdd/ReadVariableOp?dense_143/MatMul/ReadVariableOp?"main_output/BiasAdd/ReadVariableOp?!main_output/MatMul/ReadVariableOp?=token_and_position_embedding_15/embedding_30/embedding_lookup?=token_and_position_embedding_15/embedding_31/embedding_lookup?Dtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp?Htransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp?Dtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp?Htransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp?Rtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp?Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?Rtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp?Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?Rtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp?Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?Rtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp?Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?Ctransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp?Etransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp?Ctransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp?Etransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp?
%token_and_position_embedding_15/ShapeShapeinputs_0*
T0*
_output_shapes
:2'
%token_and_position_embedding_15/Shape?
3token_and_position_embedding_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3token_and_position_embedding_15/strided_slice/stack?
5token_and_position_embedding_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5token_and_position_embedding_15/strided_slice/stack_1?
5token_and_position_embedding_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5token_and_position_embedding_15/strided_slice/stack_2?
-token_and_position_embedding_15/strided_sliceStridedSlice.token_and_position_embedding_15/Shape:output:0<token_and_position_embedding_15/strided_slice/stack:output:0>token_and_position_embedding_15/strided_slice/stack_1:output:0>token_and_position_embedding_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-token_and_position_embedding_15/strided_slice?
+token_and_position_embedding_15/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2-
+token_and_position_embedding_15/range/start?
+token_and_position_embedding_15/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2-
+token_and_position_embedding_15/range/delta?
%token_and_position_embedding_15/rangeRange4token_and_position_embedding_15/range/start:output:06token_and_position_embedding_15/strided_slice:output:04token_and_position_embedding_15/range/delta:output:0*#
_output_shapes
:?????????2'
%token_and_position_embedding_15/range?
=token_and_position_embedding_15/embedding_31/embedding_lookupResourceGatherEtoken_and_position_embedding_15_embedding_31_embedding_lookup_6976567.token_and_position_embedding_15/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*X
_classN
LJloc:@token_and_position_embedding_15/embedding_31/embedding_lookup/6976567*'
_output_shapes
:????????? *
dtype02?
=token_and_position_embedding_15/embedding_31/embedding_lookup?
Ftoken_and_position_embedding_15/embedding_31/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_15/embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@token_and_position_embedding_15/embedding_31/embedding_lookup/6976567*'
_output_shapes
:????????? 2H
Ftoken_and_position_embedding_15/embedding_31/embedding_lookup/Identity?
Htoken_and_position_embedding_15/embedding_31/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_15/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2J
Htoken_and_position_embedding_15/embedding_31/embedding_lookup/Identity_1?
1token_and_position_embedding_15/embedding_30/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????(23
1token_and_position_embedding_15/embedding_30/Cast?
=token_and_position_embedding_15/embedding_30/embedding_lookupResourceGatherEtoken_and_position_embedding_15_embedding_30_embedding_lookup_69765735token_and_position_embedding_15/embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*X
_classN
LJloc:@token_and_position_embedding_15/embedding_30/embedding_lookup/6976573*+
_output_shapes
:?????????( *
dtype02?
=token_and_position_embedding_15/embedding_30/embedding_lookup?
Ftoken_and_position_embedding_15/embedding_30/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_15/embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@token_and_position_embedding_15/embedding_30/embedding_lookup/6976573*+
_output_shapes
:?????????( 2H
Ftoken_and_position_embedding_15/embedding_30/embedding_lookup/Identity?
Htoken_and_position_embedding_15/embedding_30/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_15/embedding_30/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2J
Htoken_and_position_embedding_15/embedding_30/embedding_lookup/Identity_1?
#token_and_position_embedding_15/addAddV2Qtoken_and_position_embedding_15/embedding_30/embedding_lookup/Identity_1:output:0Qtoken_and_position_embedding_15/embedding_31/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2%
#token_and_position_embedding_15/add?
7transformer_block_15/multi_head_self_attention_15/ShapeShape'token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:29
7transformer_block_15/multi_head_self_attention_15/Shape?
Etransformer_block_15/multi_head_self_attention_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block_15/multi_head_self_attention_15/strided_slice/stack?
Gtransformer_block_15/multi_head_self_attention_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_15/multi_head_self_attention_15/strided_slice/stack_1?
Gtransformer_block_15/multi_head_self_attention_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_15/multi_head_self_attention_15/strided_slice/stack_2?
?transformer_block_15/multi_head_self_attention_15/strided_sliceStridedSlice@transformer_block_15/multi_head_self_attention_15/Shape:output:0Ntransformer_block_15/multi_head_self_attention_15/strided_slice/stack:output:0Ptransformer_block_15/multi_head_self_attention_15/strided_slice/stack_1:output:0Ptransformer_block_15/multi_head_self_attention_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?transformer_block_15/multi_head_self_attention_15/strided_slice?
Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpReadVariableOp]transformer_block_15_multi_head_self_attention_15_dense_135_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02V
Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axes?
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/free?
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ShapeShape'token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Shape?
Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/free:output:0\transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2?
Utransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Utransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis?
Ptransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axes:output:0^transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Ptransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1?
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const?
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ProdProdWtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod?
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_1?
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod_1ProdYtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod_1?
Qtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat/axis?
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concatConcatV2Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/free:output:0Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axes:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat?
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/stackPackStransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod:output:0Utransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/stack?
Otransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/transpose	Transpose'token_and_position_embedding_15/add:z:0Utransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2Q
Otransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/transpose?
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReshapeReshapeStransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/transpose:y:0Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Reshape?
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/MatMulMatMulVtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Reshape:output:0\transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/MatMul?
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_2?
Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1ConcatV2Wtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_2:output:0\transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1?
Etransformer_block_15/multi_head_self_attention_15/dense_135/TensordotReshapeVtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/MatMul:product:0Wtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2G
Etransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot?
Rtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpReadVariableOp[transformer_block_15_multi_head_self_attention_15_dense_135_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp?
Ctransformer_block_15/multi_head_self_attention_15/dense_135/BiasAddBiasAddNtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2E
Ctransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd?
Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpReadVariableOp]transformer_block_15_multi_head_self_attention_15_dense_136_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02V
Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axes?
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/free?
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ShapeShape'token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Shape?
Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/free:output:0\transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2?
Utransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Utransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis?
Ptransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axes:output:0^transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Ptransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1?
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const?
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ProdProdWtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod?
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_1?
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod_1ProdYtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod_1?
Qtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat/axis?
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concatConcatV2Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/free:output:0Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axes:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat?
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/stackPackStransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod:output:0Utransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/stack?
Otransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/transpose	Transpose'token_and_position_embedding_15/add:z:0Utransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2Q
Otransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/transpose?
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReshapeReshapeStransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/transpose:y:0Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Reshape?
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/MatMulMatMulVtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Reshape:output:0\transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/MatMul?
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_2?
Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1ConcatV2Wtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_2:output:0\transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1?
Etransformer_block_15/multi_head_self_attention_15/dense_136/TensordotReshapeVtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/MatMul:product:0Wtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2G
Etransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot?
Rtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpReadVariableOp[transformer_block_15_multi_head_self_attention_15_dense_136_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp?
Ctransformer_block_15/multi_head_self_attention_15/dense_136/BiasAddBiasAddNtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2E
Ctransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd?
Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpReadVariableOp]transformer_block_15_multi_head_self_attention_15_dense_137_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02V
Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axes?
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/free?
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ShapeShape'token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Shape?
Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/free:output:0\transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2?
Utransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Utransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis?
Ptransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axes:output:0^transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Ptransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1?
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const?
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ProdProdWtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod?
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_1?
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod_1ProdYtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod_1?
Qtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat/axis?
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concatConcatV2Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/free:output:0Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axes:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat?
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/stackPackStransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod:output:0Utransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/stack?
Otransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/transpose	Transpose'token_and_position_embedding_15/add:z:0Utransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2Q
Otransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/transpose?
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReshapeReshapeStransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/transpose:y:0Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Reshape?
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/MatMulMatMulVtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Reshape:output:0\transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/MatMul?
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_2?
Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1ConcatV2Wtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_2:output:0\transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1?
Etransformer_block_15/multi_head_self_attention_15/dense_137/TensordotReshapeVtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/MatMul:product:0Wtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2G
Etransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot?
Rtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpReadVariableOp[transformer_block_15_multi_head_self_attention_15_dense_137_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp?
Ctransformer_block_15/multi_head_self_attention_15/dense_137/BiasAddBiasAddNtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2E
Ctransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd?
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/1?
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/2?
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/3?
?transformer_block_15/multi_head_self_attention_15/Reshape/shapePackHtransformer_block_15/multi_head_self_attention_15/strided_slice:output:0Jtransformer_block_15/multi_head_self_attention_15/Reshape/shape/1:output:0Jtransformer_block_15/multi_head_self_attention_15/Reshape/shape/2:output:0Jtransformer_block_15/multi_head_self_attention_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_15/multi_head_self_attention_15/Reshape/shape?
9transformer_block_15/multi_head_self_attention_15/ReshapeReshapeLtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd:output:0Htransformer_block_15/multi_head_self_attention_15/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_15/multi_head_self_attention_15/Reshape?
@transformer_block_15/multi_head_self_attention_15/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_15/multi_head_self_attention_15/transpose/perm?
;transformer_block_15/multi_head_self_attention_15/transpose	TransposeBtransformer_block_15/multi_head_self_attention_15/Reshape:output:0Itransformer_block_15/multi_head_self_attention_15/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_15/multi_head_self_attention_15/transpose?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/1?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/2?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/3?
Atransformer_block_15/multi_head_self_attention_15/Reshape_1/shapePackHtransformer_block_15/multi_head_self_attention_15/strided_slice:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/1:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/2:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2C
Atransformer_block_15/multi_head_self_attention_15/Reshape_1/shape?
;transformer_block_15/multi_head_self_attention_15/Reshape_1ReshapeLtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd:output:0Jtransformer_block_15/multi_head_self_attention_15/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_15/multi_head_self_attention_15/Reshape_1?
Btransformer_block_15/multi_head_self_attention_15/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2D
Btransformer_block_15/multi_head_self_attention_15/transpose_1/perm?
=transformer_block_15/multi_head_self_attention_15/transpose_1	TransposeDtransformer_block_15/multi_head_self_attention_15/Reshape_1:output:0Ktransformer_block_15/multi_head_self_attention_15/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2?
=transformer_block_15/multi_head_self_attention_15/transpose_1?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/1?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/2?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/3?
Atransformer_block_15/multi_head_self_attention_15/Reshape_2/shapePackHtransformer_block_15/multi_head_self_attention_15/strided_slice:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/1:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/2:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2C
Atransformer_block_15/multi_head_self_attention_15/Reshape_2/shape?
;transformer_block_15/multi_head_self_attention_15/Reshape_2ReshapeLtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd:output:0Jtransformer_block_15/multi_head_self_attention_15/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_15/multi_head_self_attention_15/Reshape_2?
Btransformer_block_15/multi_head_self_attention_15/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2D
Btransformer_block_15/multi_head_self_attention_15/transpose_2/perm?
=transformer_block_15/multi_head_self_attention_15/transpose_2	TransposeDtransformer_block_15/multi_head_self_attention_15/Reshape_2:output:0Ktransformer_block_15/multi_head_self_attention_15/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2?
=transformer_block_15/multi_head_self_attention_15/transpose_2?
8transformer_block_15/multi_head_self_attention_15/MatMulBatchMatMulV2?transformer_block_15/multi_head_self_attention_15/transpose:y:0Atransformer_block_15/multi_head_self_attention_15/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2:
8transformer_block_15/multi_head_self_attention_15/MatMul?
9transformer_block_15/multi_head_self_attention_15/Shape_1ShapeAtransformer_block_15/multi_head_self_attention_15/transpose_1:y:0*
T0*
_output_shapes
:2;
9transformer_block_15/multi_head_self_attention_15/Shape_1?
Gtransformer_block_15/multi_head_self_attention_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2I
Gtransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack?
Itransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2K
Itransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_1?
Itransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_2?
Atransformer_block_15/multi_head_self_attention_15/strided_slice_1StridedSliceBtransformer_block_15/multi_head_self_attention_15/Shape_1:output:0Ptransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack:output:0Rtransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_1:output:0Rtransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Atransformer_block_15/multi_head_self_attention_15/strided_slice_1?
6transformer_block_15/multi_head_self_attention_15/CastCastJtransformer_block_15/multi_head_self_attention_15/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 28
6transformer_block_15/multi_head_self_attention_15/Cast?
6transformer_block_15/multi_head_self_attention_15/SqrtSqrt:transformer_block_15/multi_head_self_attention_15/Cast:y:0*
T0*
_output_shapes
: 28
6transformer_block_15/multi_head_self_attention_15/Sqrt?
9transformer_block_15/multi_head_self_attention_15/truedivRealDivAtransformer_block_15/multi_head_self_attention_15/MatMul:output:0:transformer_block_15/multi_head_self_attention_15/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2;
9transformer_block_15/multi_head_self_attention_15/truediv?
9transformer_block_15/multi_head_self_attention_15/SoftmaxSoftmax=transformer_block_15/multi_head_self_attention_15/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2;
9transformer_block_15/multi_head_self_attention_15/Softmax?
:transformer_block_15/multi_head_self_attention_15/MatMul_1BatchMatMulV2Ctransformer_block_15/multi_head_self_attention_15/Softmax:softmax:0Atransformer_block_15/multi_head_self_attention_15/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2<
:transformer_block_15/multi_head_self_attention_15/MatMul_1?
Btransformer_block_15/multi_head_self_attention_15/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2D
Btransformer_block_15/multi_head_self_attention_15/transpose_3/perm?
=transformer_block_15/multi_head_self_attention_15/transpose_3	TransposeCtransformer_block_15/multi_head_self_attention_15/MatMul_1:output:0Ktransformer_block_15/multi_head_self_attention_15/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2?
=transformer_block_15/multi_head_self_attention_15/transpose_3?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/1?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/2?
Atransformer_block_15/multi_head_self_attention_15/Reshape_3/shapePackHtransformer_block_15/multi_head_self_attention_15/strided_slice:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/1:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2C
Atransformer_block_15/multi_head_self_attention_15/Reshape_3/shape?
;transformer_block_15/multi_head_self_attention_15/Reshape_3ReshapeAtransformer_block_15/multi_head_self_attention_15/transpose_3:y:0Jtransformer_block_15/multi_head_self_attention_15/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2=
;transformer_block_15/multi_head_self_attention_15/Reshape_3?
Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpReadVariableOp]transformer_block_15_multi_head_self_attention_15_dense_138_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02V
Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axes?
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/free?
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ShapeShapeDtransformer_block_15/multi_head_self_attention_15/Reshape_3:output:0*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Shape?
Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/free:output:0\transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2?
Utransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Utransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis?
Ptransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axes:output:0^transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Ptransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1?
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const?
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ProdProdWtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod?
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_1?
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod_1ProdYtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod_1?
Qtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat/axis?
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concatConcatV2Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/free:output:0Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axes:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat?
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/stackPackStransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod:output:0Utransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/stack?
Otransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/transpose	TransposeDtransformer_block_15/multi_head_self_attention_15/Reshape_3:output:0Utransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2Q
Otransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/transpose?
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReshapeReshapeStransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/transpose:y:0Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Reshape?
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/MatMulMatMulVtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Reshape:output:0\transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/MatMul?
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_2?
Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1ConcatV2Wtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_2:output:0\transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1?
Etransformer_block_15/multi_head_self_attention_15/dense_138/TensordotReshapeVtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/MatMul:product:0Wtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 2G
Etransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot?
Rtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpReadVariableOp[transformer_block_15_multi_head_self_attention_15_dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp?
Ctransformer_block_15/multi_head_self_attention_15/dense_138/BiasAddBiasAddNtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2E
Ctransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd?
(transformer_block_15/dropout_30/IdentityIdentityLtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 2*
(transformer_block_15/dropout_30/Identity?
transformer_block_15/addAddV2'token_and_position_embedding_15/add:z:01transformer_block_15/dropout_30/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_15/add?
Jtransformer_block_15/layer_normalization_30/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/layer_normalization_30/moments/mean/reduction_indices?
8transformer_block_15/layer_normalization_30/moments/meanMeantransformer_block_15/add:z:0Stransformer_block_15/layer_normalization_30/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2:
8transformer_block_15/layer_normalization_30/moments/mean?
@transformer_block_15/layer_normalization_30/moments/StopGradientStopGradientAtransformer_block_15/layer_normalization_30/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2B
@transformer_block_15/layer_normalization_30/moments/StopGradient?
Etransformer_block_15/layer_normalization_30/moments/SquaredDifferenceSquaredDifferencetransformer_block_15/add:z:0Itransformer_block_15/layer_normalization_30/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2G
Etransformer_block_15/layer_normalization_30/moments/SquaredDifference?
Ntransformer_block_15/layer_normalization_30/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Ntransformer_block_15/layer_normalization_30/moments/variance/reduction_indices?
<transformer_block_15/layer_normalization_30/moments/varianceMeanItransformer_block_15/layer_normalization_30/moments/SquaredDifference:z:0Wtransformer_block_15/layer_normalization_30/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2>
<transformer_block_15/layer_normalization_30/moments/variance?
;transformer_block_15/layer_normalization_30/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52=
;transformer_block_15/layer_normalization_30/batchnorm/add/y?
9transformer_block_15/layer_normalization_30/batchnorm/addAddV2Etransformer_block_15/layer_normalization_30/moments/variance:output:0Dtransformer_block_15/layer_normalization_30/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2;
9transformer_block_15/layer_normalization_30/batchnorm/add?
;transformer_block_15/layer_normalization_30/batchnorm/RsqrtRsqrt=transformer_block_15/layer_normalization_30/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2=
;transformer_block_15/layer_normalization_30/batchnorm/Rsqrt?
Htransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_block_15_layer_normalization_30_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02J
Htransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp?
9transformer_block_15/layer_normalization_30/batchnorm/mulMul?transformer_block_15/layer_normalization_30/batchnorm/Rsqrt:y:0Ptransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_15/layer_normalization_30/batchnorm/mul?
;transformer_block_15/layer_normalization_30/batchnorm/mul_1Multransformer_block_15/add:z:0=transformer_block_15/layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_30/batchnorm/mul_1?
;transformer_block_15/layer_normalization_30/batchnorm/mul_2MulAtransformer_block_15/layer_normalization_30/moments/mean:output:0=transformer_block_15/layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_30/batchnorm/mul_2?
Dtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOpReadVariableOpMtransformer_block_15_layer_normalization_30_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02F
Dtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp?
9transformer_block_15/layer_normalization_30/batchnorm/subSubLtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp:value:0?transformer_block_15/layer_normalization_30/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_15/layer_normalization_30/batchnorm/sub?
;transformer_block_15/layer_normalization_30/batchnorm/add_1AddV2?transformer_block_15/layer_normalization_30/batchnorm/mul_1:z:0=transformer_block_15/layer_normalization_30/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_30/batchnorm/add_1?
Etransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOpReadVariableOpNtransformer_block_15_sequential_15_dense_139_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02G
Etransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp?
;transformer_block_15/sequential_15/dense_139/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block_15/sequential_15/dense_139/Tensordot/axes?
;transformer_block_15/sequential_15/dense_139/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2=
;transformer_block_15/sequential_15/dense_139/Tensordot/free?
<transformer_block_15/sequential_15/dense_139/Tensordot/ShapeShape?transformer_block_15/layer_normalization_30/batchnorm/add_1:z:0*
T0*
_output_shapes
:2>
<transformer_block_15/sequential_15/dense_139/Tensordot/Shape?
Dtransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2/axis?
?transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2GatherV2Etransformer_block_15/sequential_15/dense_139/Tensordot/Shape:output:0Dtransformer_block_15/sequential_15/dense_139/Tensordot/free:output:0Mtransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2A
?transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2?
Ftransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Ftransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1/axis?
Atransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1GatherV2Etransformer_block_15/sequential_15/dense_139/Tensordot/Shape:output:0Dtransformer_block_15/sequential_15/dense_139/Tensordot/axes:output:0Otransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Atransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1?
<transformer_block_15/sequential_15/dense_139/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<transformer_block_15/sequential_15/dense_139/Tensordot/Const?
;transformer_block_15/sequential_15/dense_139/Tensordot/ProdProdHtransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2:output:0Etransformer_block_15/sequential_15/dense_139/Tensordot/Const:output:0*
T0*
_output_shapes
: 2=
;transformer_block_15/sequential_15/dense_139/Tensordot/Prod?
>transformer_block_15/sequential_15/dense_139/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>transformer_block_15/sequential_15/dense_139/Tensordot/Const_1?
=transformer_block_15/sequential_15/dense_139/Tensordot/Prod_1ProdJtransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1:output:0Gtransformer_block_15/sequential_15/dense_139/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2?
=transformer_block_15/sequential_15/dense_139/Tensordot/Prod_1?
Btransformer_block_15/sequential_15/dense_139/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_15/sequential_15/dense_139/Tensordot/concat/axis?
=transformer_block_15/sequential_15/dense_139/Tensordot/concatConcatV2Dtransformer_block_15/sequential_15/dense_139/Tensordot/free:output:0Dtransformer_block_15/sequential_15/dense_139/Tensordot/axes:output:0Ktransformer_block_15/sequential_15/dense_139/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_15/sequential_15/dense_139/Tensordot/concat?
<transformer_block_15/sequential_15/dense_139/Tensordot/stackPackDtransformer_block_15/sequential_15/dense_139/Tensordot/Prod:output:0Ftransformer_block_15/sequential_15/dense_139/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_15/sequential_15/dense_139/Tensordot/stack?
@transformer_block_15/sequential_15/dense_139/Tensordot/transpose	Transpose?transformer_block_15/layer_normalization_30/batchnorm/add_1:z:0Ftransformer_block_15/sequential_15/dense_139/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_15/sequential_15/dense_139/Tensordot/transpose?
>transformer_block_15/sequential_15/dense_139/Tensordot/ReshapeReshapeDtransformer_block_15/sequential_15/dense_139/Tensordot/transpose:y:0Etransformer_block_15/sequential_15/dense_139/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2@
>transformer_block_15/sequential_15/dense_139/Tensordot/Reshape?
=transformer_block_15/sequential_15/dense_139/Tensordot/MatMulMatMulGtransformer_block_15/sequential_15/dense_139/Tensordot/Reshape:output:0Mtransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2?
=transformer_block_15/sequential_15/dense_139/Tensordot/MatMul?
>transformer_block_15/sequential_15/dense_139/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2@
>transformer_block_15/sequential_15/dense_139/Tensordot/Const_2?
Dtransformer_block_15/sequential_15/dense_139/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_15/sequential_15/dense_139/Tensordot/concat_1/axis?
?transformer_block_15/sequential_15/dense_139/Tensordot/concat_1ConcatV2Htransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2:output:0Gtransformer_block_15/sequential_15/dense_139/Tensordot/Const_2:output:0Mtransformer_block_15/sequential_15/dense_139/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_15/sequential_15/dense_139/Tensordot/concat_1?
6transformer_block_15/sequential_15/dense_139/TensordotReshapeGtransformer_block_15/sequential_15/dense_139/Tensordot/MatMul:product:0Htransformer_block_15/sequential_15/dense_139/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 28
6transformer_block_15/sequential_15/dense_139/Tensordot?
Ctransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOpReadVariableOpLtransformer_block_15_sequential_15_dense_139_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp?
4transformer_block_15/sequential_15/dense_139/BiasAddBiasAdd?transformer_block_15/sequential_15/dense_139/Tensordot:output:0Ktransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 26
4transformer_block_15/sequential_15/dense_139/BiasAdd?
1transformer_block_15/sequential_15/dense_139/ReluRelu=transformer_block_15/sequential_15/dense_139/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 23
1transformer_block_15/sequential_15/dense_139/Relu?
Etransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOpReadVariableOpNtransformer_block_15_sequential_15_dense_140_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02G
Etransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp?
;transformer_block_15/sequential_15/dense_140/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block_15/sequential_15/dense_140/Tensordot/axes?
;transformer_block_15/sequential_15/dense_140/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2=
;transformer_block_15/sequential_15/dense_140/Tensordot/free?
<transformer_block_15/sequential_15/dense_140/Tensordot/ShapeShape?transformer_block_15/sequential_15/dense_139/Relu:activations:0*
T0*
_output_shapes
:2>
<transformer_block_15/sequential_15/dense_140/Tensordot/Shape?
Dtransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2/axis?
?transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2GatherV2Etransformer_block_15/sequential_15/dense_140/Tensordot/Shape:output:0Dtransformer_block_15/sequential_15/dense_140/Tensordot/free:output:0Mtransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2A
?transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2?
Ftransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Ftransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1/axis?
Atransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1GatherV2Etransformer_block_15/sequential_15/dense_140/Tensordot/Shape:output:0Dtransformer_block_15/sequential_15/dense_140/Tensordot/axes:output:0Otransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Atransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1?
<transformer_block_15/sequential_15/dense_140/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<transformer_block_15/sequential_15/dense_140/Tensordot/Const?
;transformer_block_15/sequential_15/dense_140/Tensordot/ProdProdHtransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2:output:0Etransformer_block_15/sequential_15/dense_140/Tensordot/Const:output:0*
T0*
_output_shapes
: 2=
;transformer_block_15/sequential_15/dense_140/Tensordot/Prod?
>transformer_block_15/sequential_15/dense_140/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>transformer_block_15/sequential_15/dense_140/Tensordot/Const_1?
=transformer_block_15/sequential_15/dense_140/Tensordot/Prod_1ProdJtransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1:output:0Gtransformer_block_15/sequential_15/dense_140/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2?
=transformer_block_15/sequential_15/dense_140/Tensordot/Prod_1?
Btransformer_block_15/sequential_15/dense_140/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_15/sequential_15/dense_140/Tensordot/concat/axis?
=transformer_block_15/sequential_15/dense_140/Tensordot/concatConcatV2Dtransformer_block_15/sequential_15/dense_140/Tensordot/free:output:0Dtransformer_block_15/sequential_15/dense_140/Tensordot/axes:output:0Ktransformer_block_15/sequential_15/dense_140/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_15/sequential_15/dense_140/Tensordot/concat?
<transformer_block_15/sequential_15/dense_140/Tensordot/stackPackDtransformer_block_15/sequential_15/dense_140/Tensordot/Prod:output:0Ftransformer_block_15/sequential_15/dense_140/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_15/sequential_15/dense_140/Tensordot/stack?
@transformer_block_15/sequential_15/dense_140/Tensordot/transpose	Transpose?transformer_block_15/sequential_15/dense_139/Relu:activations:0Ftransformer_block_15/sequential_15/dense_140/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_15/sequential_15/dense_140/Tensordot/transpose?
>transformer_block_15/sequential_15/dense_140/Tensordot/ReshapeReshapeDtransformer_block_15/sequential_15/dense_140/Tensordot/transpose:y:0Etransformer_block_15/sequential_15/dense_140/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2@
>transformer_block_15/sequential_15/dense_140/Tensordot/Reshape?
=transformer_block_15/sequential_15/dense_140/Tensordot/MatMulMatMulGtransformer_block_15/sequential_15/dense_140/Tensordot/Reshape:output:0Mtransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2?
=transformer_block_15/sequential_15/dense_140/Tensordot/MatMul?
>transformer_block_15/sequential_15/dense_140/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2@
>transformer_block_15/sequential_15/dense_140/Tensordot/Const_2?
Dtransformer_block_15/sequential_15/dense_140/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_15/sequential_15/dense_140/Tensordot/concat_1/axis?
?transformer_block_15/sequential_15/dense_140/Tensordot/concat_1ConcatV2Htransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2:output:0Gtransformer_block_15/sequential_15/dense_140/Tensordot/Const_2:output:0Mtransformer_block_15/sequential_15/dense_140/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_15/sequential_15/dense_140/Tensordot/concat_1?
6transformer_block_15/sequential_15/dense_140/TensordotReshapeGtransformer_block_15/sequential_15/dense_140/Tensordot/MatMul:product:0Htransformer_block_15/sequential_15/dense_140/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 28
6transformer_block_15/sequential_15/dense_140/Tensordot?
Ctransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOpReadVariableOpLtransformer_block_15_sequential_15_dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp?
4transformer_block_15/sequential_15/dense_140/BiasAddBiasAdd?transformer_block_15/sequential_15/dense_140/Tensordot:output:0Ktransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 26
4transformer_block_15/sequential_15/dense_140/BiasAdd?
(transformer_block_15/dropout_31/IdentityIdentity=transformer_block_15/sequential_15/dense_140/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2*
(transformer_block_15/dropout_31/Identity?
transformer_block_15/add_1AddV2?transformer_block_15/layer_normalization_30/batchnorm/add_1:z:01transformer_block_15/dropout_31/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_15/add_1?
Jtransformer_block_15/layer_normalization_31/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/layer_normalization_31/moments/mean/reduction_indices?
8transformer_block_15/layer_normalization_31/moments/meanMeantransformer_block_15/add_1:z:0Stransformer_block_15/layer_normalization_31/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2:
8transformer_block_15/layer_normalization_31/moments/mean?
@transformer_block_15/layer_normalization_31/moments/StopGradientStopGradientAtransformer_block_15/layer_normalization_31/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2B
@transformer_block_15/layer_normalization_31/moments/StopGradient?
Etransformer_block_15/layer_normalization_31/moments/SquaredDifferenceSquaredDifferencetransformer_block_15/add_1:z:0Itransformer_block_15/layer_normalization_31/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2G
Etransformer_block_15/layer_normalization_31/moments/SquaredDifference?
Ntransformer_block_15/layer_normalization_31/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Ntransformer_block_15/layer_normalization_31/moments/variance/reduction_indices?
<transformer_block_15/layer_normalization_31/moments/varianceMeanItransformer_block_15/layer_normalization_31/moments/SquaredDifference:z:0Wtransformer_block_15/layer_normalization_31/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2>
<transformer_block_15/layer_normalization_31/moments/variance?
;transformer_block_15/layer_normalization_31/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52=
;transformer_block_15/layer_normalization_31/batchnorm/add/y?
9transformer_block_15/layer_normalization_31/batchnorm/addAddV2Etransformer_block_15/layer_normalization_31/moments/variance:output:0Dtransformer_block_15/layer_normalization_31/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2;
9transformer_block_15/layer_normalization_31/batchnorm/add?
;transformer_block_15/layer_normalization_31/batchnorm/RsqrtRsqrt=transformer_block_15/layer_normalization_31/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2=
;transformer_block_15/layer_normalization_31/batchnorm/Rsqrt?
Htransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_block_15_layer_normalization_31_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02J
Htransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp?
9transformer_block_15/layer_normalization_31/batchnorm/mulMul?transformer_block_15/layer_normalization_31/batchnorm/Rsqrt:y:0Ptransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_15/layer_normalization_31/batchnorm/mul?
;transformer_block_15/layer_normalization_31/batchnorm/mul_1Multransformer_block_15/add_1:z:0=transformer_block_15/layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_31/batchnorm/mul_1?
;transformer_block_15/layer_normalization_31/batchnorm/mul_2MulAtransformer_block_15/layer_normalization_31/moments/mean:output:0=transformer_block_15/layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_31/batchnorm/mul_2?
Dtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOpReadVariableOpMtransformer_block_15_layer_normalization_31_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02F
Dtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp?
9transformer_block_15/layer_normalization_31/batchnorm/subSubLtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp:value:0?transformer_block_15/layer_normalization_31/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_15/layer_normalization_31/batchnorm/sub?
;transformer_block_15/layer_normalization_31/batchnorm/add_1AddV2?transformer_block_15/layer_normalization_31/batchnorm/mul_1:z:0=transformer_block_15/layer_normalization_31/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_31/batchnorm/add_1?
2global_average_pooling1d_15/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2global_average_pooling1d_15/Mean/reduction_indices?
 global_average_pooling1d_15/MeanMean?transformer_block_15/layer_normalization_31/batchnorm/add_1:z:0;global_average_pooling1d_15/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2"
 global_average_pooling1d_15/Mean?
 aux_output/MatMul/ReadVariableOpReadVariableOp)aux_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 aux_output/MatMul/ReadVariableOp?
aux_output/MatMulMatMul)global_average_pooling1d_15/Mean:output:0(aux_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
aux_output/MatMul?
!aux_output/BiasAdd/ReadVariableOpReadVariableOp*aux_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!aux_output/BiasAdd/ReadVariableOp?
aux_output/BiasAddBiasAddaux_output/MatMul:product:0)aux_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
aux_output/BiasAdd?
aux_output/SigmoidSigmoidaux_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
aux_output/Sigmoidz
concatenate_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_15/concat/axis?
concatenate_15/concatConcatV2aux_output/Sigmoid:y:0inputs_1inputs_2#concatenate_15/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_15/concat?
dense_141/MatMul/ReadVariableOpReadVariableOp(dense_141_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_141/MatMul/ReadVariableOp?
dense_141/MatMulMatMulconcatenate_15/concat:output:0'dense_141/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_141/MatMul?
 dense_141/BiasAdd/ReadVariableOpReadVariableOp)dense_141_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_141/BiasAdd/ReadVariableOp?
dense_141/BiasAddBiasAdddense_141/MatMul:product:0(dense_141/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_141/BiasAddv
dense_141/ReluReludense_141/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_141/Relu?
dense_142/MatMul/ReadVariableOpReadVariableOp(dense_142_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_142/MatMul/ReadVariableOp?
dense_142/MatMulMatMuldense_141/Relu:activations:0'dense_142/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_142/MatMul?
 dense_142/BiasAdd/ReadVariableOpReadVariableOp)dense_142_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_142/BiasAdd/ReadVariableOp?
dense_142/BiasAddBiasAdddense_142/MatMul:product:0(dense_142/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_142/BiasAddv
dense_142/ReluReludense_142/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_142/Relu?
dense_143/MatMul/ReadVariableOpReadVariableOp(dense_143_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_143/MatMul/ReadVariableOp?
dense_143/MatMulMatMuldense_142/Relu:activations:0'dense_143/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_143/MatMul?
 dense_143/BiasAdd/ReadVariableOpReadVariableOp)dense_143_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_143/BiasAdd/ReadVariableOp?
dense_143/BiasAddBiasAdddense_143/MatMul:product:0(dense_143/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_143/BiasAddv
dense_143/ReluReludense_143/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_143/Relu?
!main_output/MatMul/ReadVariableOpReadVariableOp*main_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!main_output/MatMul/ReadVariableOp?
main_output/MatMulMatMuldense_143/Relu:activations:0)main_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
main_output/MatMul?
"main_output/BiasAdd/ReadVariableOpReadVariableOp+main_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"main_output/BiasAdd/ReadVariableOp?
main_output/BiasAddBiasAddmain_output/MatMul:product:0*main_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
main_output/BiasAdd?
main_output/SigmoidSigmoidmain_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
main_output/Sigmoidr
IdentityIdentitymain_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityu

Identity_1Identityaux_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp"^aux_output/BiasAdd/ReadVariableOp!^aux_output/MatMul/ReadVariableOp!^dense_141/BiasAdd/ReadVariableOp ^dense_141/MatMul/ReadVariableOp!^dense_142/BiasAdd/ReadVariableOp ^dense_142/MatMul/ReadVariableOp!^dense_143/BiasAdd/ReadVariableOp ^dense_143/MatMul/ReadVariableOp#^main_output/BiasAdd/ReadVariableOp"^main_output/MatMul/ReadVariableOp>^token_and_position_embedding_15/embedding_30/embedding_lookup>^token_and_position_embedding_15/embedding_31/embedding_lookupE^transformer_block_15/layer_normalization_30/batchnorm/ReadVariableOpI^transformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOpE^transformer_block_15/layer_normalization_31/batchnorm/ReadVariableOpI^transformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOpS^transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpU^transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpS^transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpU^transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpS^transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpU^transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpS^transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpU^transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpD^transformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOpF^transformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOpD^transformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOpF^transformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!aux_output/BiasAdd/ReadVariableOp!aux_output/BiasAdd/ReadVariableOp2D
 aux_output/MatMul/ReadVariableOp aux_output/MatMul/ReadVariableOp2D
 dense_141/BiasAdd/ReadVariableOp dense_141/BiasAdd/ReadVariableOp2B
dense_141/MatMul/ReadVariableOpdense_141/MatMul/ReadVariableOp2D
 dense_142/BiasAdd/ReadVariableOp dense_142/BiasAdd/ReadVariableOp2B
dense_142/MatMul/ReadVariableOpdense_142/MatMul/ReadVariableOp2D
 dense_143/BiasAdd/ReadVariableOp dense_143/BiasAdd/ReadVariableOp2B
dense_143/MatMul/ReadVariableOpdense_143/MatMul/ReadVariableOp2H
"main_output/BiasAdd/ReadVariableOp"main_output/BiasAdd/ReadVariableOp2F
!main_output/MatMul/ReadVariableOp!main_output/MatMul/ReadVariableOp2~
=token_and_position_embedding_15/embedding_30/embedding_lookup=token_and_position_embedding_15/embedding_30/embedding_lookup2~
=token_and_position_embedding_15/embedding_31/embedding_lookup=token_and_position_embedding_15/embedding_31/embedding_lookup2?
Dtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOpDtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp2?
Htransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOpHtransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp2?
Dtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOpDtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp2?
Htransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOpHtransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp2?
Rtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpRtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp2?
Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp2?
Rtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpRtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp2?
Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp2?
Rtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpRtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp2?
Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp2?
Rtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpRtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp2?
Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp2?
Ctransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOpCtransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp2?
Etransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOpEtransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp2?
Ctransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOpCtransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp2?
Etransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOpEtransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
??
?
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_6975465

inputsZ
Hmulti_head_self_attention_15_dense_135_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_135_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_136_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_136_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_137_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_137_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_138_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_138_biasadd_readvariableop_resource: J
<layer_normalization_30_batchnorm_mul_readvariableop_resource: F
8layer_normalization_30_batchnorm_readvariableop_resource: K
9sequential_15_dense_139_tensordot_readvariableop_resource:  E
7sequential_15_dense_139_biasadd_readvariableop_resource: K
9sequential_15_dense_140_tensordot_readvariableop_resource:  E
7sequential_15_dense_140_biasadd_readvariableop_resource: J
<layer_normalization_31_batchnorm_mul_readvariableop_resource: F
8layer_normalization_31_batchnorm_readvariableop_resource: 
identity??/layer_normalization_30/batchnorm/ReadVariableOp?3layer_normalization_30/batchnorm/mul/ReadVariableOp?/layer_normalization_31/batchnorm/ReadVariableOp?3layer_normalization_31/batchnorm/mul/ReadVariableOp?=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?.sequential_15/dense_139/BiasAdd/ReadVariableOp?0sequential_15/dense_139/Tensordot/ReadVariableOp?.sequential_15/dense_140/BiasAdd/ReadVariableOp?0sequential_15/dense_140/Tensordot/ReadVariableOp~
"multi_head_self_attention_15/ShapeShapeinputs*
T0*
_output_shapes
:2$
"multi_head_self_attention_15/Shape?
0multi_head_self_attention_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0multi_head_self_attention_15/strided_slice/stack?
2multi_head_self_attention_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2multi_head_self_attention_15/strided_slice/stack_1?
2multi_head_self_attention_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2multi_head_self_attention_15/strided_slice/stack_2?
*multi_head_self_attention_15/strided_sliceStridedSlice+multi_head_self_attention_15/Shape:output:09multi_head_self_attention_15/strided_slice/stack:output:0;multi_head_self_attention_15/strided_slice/stack_1:output:0;multi_head_self_attention_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*multi_head_self_attention_15/strided_slice?
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_135_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_135/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_135/Tensordot/axes?
5multi_head_self_attention_15/dense_135/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_135/Tensordot/free?
6multi_head_self_attention_15/dense_135/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_135/Tensordot/Shape?
>multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_135/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_135/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_135/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_135/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_135/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_135/Tensordot/Const?
5multi_head_self_attention_15/dense_135/Tensordot/ProdProdBmulti_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_135/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_135/Tensordot/Prod?
8multi_head_self_attention_15/dense_135/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_135/Tensordot/Const_1?
7multi_head_self_attention_15/dense_135/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_135/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_135/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_135/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_135/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_135/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_135/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_135/Tensordot/free:output:0>multi_head_self_attention_15/dense_135/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_135/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_135/Tensordot/concat?
6multi_head_self_attention_15/dense_135/Tensordot/stackPack>multi_head_self_attention_15/dense_135/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_135/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_135/Tensordot/stack?
:multi_head_self_attention_15/dense_135/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_135/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_135/Tensordot/transpose?
8multi_head_self_attention_15/dense_135/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_135/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_135/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_135/Tensordot/Reshape?
7multi_head_self_attention_15/dense_135/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_135/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_135/Tensordot/MatMul?
8multi_head_self_attention_15/dense_135/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_135/Tensordot/Const_2?
>multi_head_self_attention_15/dense_135/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_135/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_135/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_135/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_135/Tensordot/concat_1?
0multi_head_self_attention_15/dense_135/TensordotReshapeAmulti_head_self_attention_15/dense_135/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_135/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_135/Tensordot?
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_135_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_135/BiasAddBiasAdd9multi_head_self_attention_15/dense_135/Tensordot:output:0Emulti_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_135/BiasAdd?
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_136_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_136/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_136/Tensordot/axes?
5multi_head_self_attention_15/dense_136/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_136/Tensordot/free?
6multi_head_self_attention_15/dense_136/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_136/Tensordot/Shape?
>multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_136/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_136/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_136/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_136/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_136/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_136/Tensordot/Const?
5multi_head_self_attention_15/dense_136/Tensordot/ProdProdBmulti_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_136/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_136/Tensordot/Prod?
8multi_head_self_attention_15/dense_136/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_136/Tensordot/Const_1?
7multi_head_self_attention_15/dense_136/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_136/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_136/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_136/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_136/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_136/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_136/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_136/Tensordot/free:output:0>multi_head_self_attention_15/dense_136/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_136/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_136/Tensordot/concat?
6multi_head_self_attention_15/dense_136/Tensordot/stackPack>multi_head_self_attention_15/dense_136/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_136/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_136/Tensordot/stack?
:multi_head_self_attention_15/dense_136/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_136/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_136/Tensordot/transpose?
8multi_head_self_attention_15/dense_136/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_136/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_136/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_136/Tensordot/Reshape?
7multi_head_self_attention_15/dense_136/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_136/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_136/Tensordot/MatMul?
8multi_head_self_attention_15/dense_136/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_136/Tensordot/Const_2?
>multi_head_self_attention_15/dense_136/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_136/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_136/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_136/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_136/Tensordot/concat_1?
0multi_head_self_attention_15/dense_136/TensordotReshapeAmulti_head_self_attention_15/dense_136/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_136/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_136/Tensordot?
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_136_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_136/BiasAddBiasAdd9multi_head_self_attention_15/dense_136/Tensordot:output:0Emulti_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_136/BiasAdd?
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_137_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_137/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_137/Tensordot/axes?
5multi_head_self_attention_15/dense_137/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_137/Tensordot/free?
6multi_head_self_attention_15/dense_137/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_137/Tensordot/Shape?
>multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_137/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_137/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_137/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_137/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_137/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_137/Tensordot/Const?
5multi_head_self_attention_15/dense_137/Tensordot/ProdProdBmulti_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_137/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_137/Tensordot/Prod?
8multi_head_self_attention_15/dense_137/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_137/Tensordot/Const_1?
7multi_head_self_attention_15/dense_137/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_137/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_137/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_137/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_137/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_137/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_137/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_137/Tensordot/free:output:0>multi_head_self_attention_15/dense_137/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_137/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_137/Tensordot/concat?
6multi_head_self_attention_15/dense_137/Tensordot/stackPack>multi_head_self_attention_15/dense_137/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_137/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_137/Tensordot/stack?
:multi_head_self_attention_15/dense_137/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_137/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_137/Tensordot/transpose?
8multi_head_self_attention_15/dense_137/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_137/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_137/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_137/Tensordot/Reshape?
7multi_head_self_attention_15/dense_137/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_137/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_137/Tensordot/MatMul?
8multi_head_self_attention_15/dense_137/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_137/Tensordot/Const_2?
>multi_head_self_attention_15/dense_137/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_137/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_137/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_137/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_137/Tensordot/concat_1?
0multi_head_self_attention_15/dense_137/TensordotReshapeAmulti_head_self_attention_15/dense_137/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_137/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_137/Tensordot?
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_137_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_137/BiasAddBiasAdd9multi_head_self_attention_15/dense_137/Tensordot:output:0Emulti_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_137/BiasAdd?
,multi_head_self_attention_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,multi_head_self_attention_15/Reshape/shape/1?
,multi_head_self_attention_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2.
,multi_head_self_attention_15/Reshape/shape/2?
,multi_head_self_attention_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2.
,multi_head_self_attention_15/Reshape/shape/3?
*multi_head_self_attention_15/Reshape/shapePack3multi_head_self_attention_15/strided_slice:output:05multi_head_self_attention_15/Reshape/shape/1:output:05multi_head_self_attention_15/Reshape/shape/2:output:05multi_head_self_attention_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2,
*multi_head_self_attention_15/Reshape/shape?
$multi_head_self_attention_15/ReshapeReshape7multi_head_self_attention_15/dense_135/BiasAdd:output:03multi_head_self_attention_15/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_15/Reshape?
+multi_head_self_attention_15/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+multi_head_self_attention_15/transpose/perm?
&multi_head_self_attention_15/transpose	Transpose-multi_head_self_attention_15/Reshape:output:04multi_head_self_attention_15/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/transpose?
.multi_head_self_attention_15/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_1/shape/1?
.multi_head_self_attention_15/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_1/shape/2?
.multi_head_self_attention_15/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_1/shape/3?
,multi_head_self_attention_15/Reshape_1/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_1/shape/1:output:07multi_head_self_attention_15/Reshape_1/shape/2:output:07multi_head_self_attention_15/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_1/shape?
&multi_head_self_attention_15/Reshape_1Reshape7multi_head_self_attention_15/dense_136/BiasAdd:output:05multi_head_self_attention_15/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/Reshape_1?
-multi_head_self_attention_15/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_1/perm?
(multi_head_self_attention_15/transpose_1	Transpose/multi_head_self_attention_15/Reshape_1:output:06multi_head_self_attention_15/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_1?
.multi_head_self_attention_15/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_2/shape/1?
.multi_head_self_attention_15/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_2/shape/2?
.multi_head_self_attention_15/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_2/shape/3?
,multi_head_self_attention_15/Reshape_2/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_2/shape/1:output:07multi_head_self_attention_15/Reshape_2/shape/2:output:07multi_head_self_attention_15/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_2/shape?
&multi_head_self_attention_15/Reshape_2Reshape7multi_head_self_attention_15/dense_137/BiasAdd:output:05multi_head_self_attention_15/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/Reshape_2?
-multi_head_self_attention_15/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_2/perm?
(multi_head_self_attention_15/transpose_2	Transpose/multi_head_self_attention_15/Reshape_2:output:06multi_head_self_attention_15/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_2?
#multi_head_self_attention_15/MatMulBatchMatMulV2*multi_head_self_attention_15/transpose:y:0,multi_head_self_attention_15/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2%
#multi_head_self_attention_15/MatMul?
$multi_head_self_attention_15/Shape_1Shape,multi_head_self_attention_15/transpose_1:y:0*
T0*
_output_shapes
:2&
$multi_head_self_attention_15/Shape_1?
2multi_head_self_attention_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2multi_head_self_attention_15/strided_slice_1/stack?
4multi_head_self_attention_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_15/strided_slice_1/stack_1?
4multi_head_self_attention_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4multi_head_self_attention_15/strided_slice_1/stack_2?
,multi_head_self_attention_15/strided_slice_1StridedSlice-multi_head_self_attention_15/Shape_1:output:0;multi_head_self_attention_15/strided_slice_1/stack:output:0=multi_head_self_attention_15/strided_slice_1/stack_1:output:0=multi_head_self_attention_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,multi_head_self_attention_15/strided_slice_1?
!multi_head_self_attention_15/CastCast5multi_head_self_attention_15/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!multi_head_self_attention_15/Cast?
!multi_head_self_attention_15/SqrtSqrt%multi_head_self_attention_15/Cast:y:0*
T0*
_output_shapes
: 2#
!multi_head_self_attention_15/Sqrt?
$multi_head_self_attention_15/truedivRealDiv,multi_head_self_attention_15/MatMul:output:0%multi_head_self_attention_15/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2&
$multi_head_self_attention_15/truediv?
$multi_head_self_attention_15/SoftmaxSoftmax(multi_head_self_attention_15/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2&
$multi_head_self_attention_15/Softmax?
%multi_head_self_attention_15/MatMul_1BatchMatMulV2.multi_head_self_attention_15/Softmax:softmax:0,multi_head_self_attention_15/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_15/MatMul_1?
-multi_head_self_attention_15/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_3/perm?
(multi_head_self_attention_15/transpose_3	Transpose.multi_head_self_attention_15/MatMul_1:output:06multi_head_self_attention_15/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_3?
.multi_head_self_attention_15/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_3/shape/1?
.multi_head_self_attention_15/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_self_attention_15/Reshape_3/shape/2?
,multi_head_self_attention_15/Reshape_3/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_3/shape/1:output:07multi_head_self_attention_15/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_3/shape?
&multi_head_self_attention_15/Reshape_3Reshape,multi_head_self_attention_15/transpose_3:y:05multi_head_self_attention_15/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2(
&multi_head_self_attention_15/Reshape_3?
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_138_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_138/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_138/Tensordot/axes?
5multi_head_self_attention_15/dense_138/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_138/Tensordot/free?
6multi_head_self_attention_15/dense_138/Tensordot/ShapeShape/multi_head_self_attention_15/Reshape_3:output:0*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_138/Tensordot/Shape?
>multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_138/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_138/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_138/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_138/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_138/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_138/Tensordot/Const?
5multi_head_self_attention_15/dense_138/Tensordot/ProdProdBmulti_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_138/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_138/Tensordot/Prod?
8multi_head_self_attention_15/dense_138/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_138/Tensordot/Const_1?
7multi_head_self_attention_15/dense_138/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_138/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_138/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_138/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_138/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_138/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_138/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_138/Tensordot/free:output:0>multi_head_self_attention_15/dense_138/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_138/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_138/Tensordot/concat?
6multi_head_self_attention_15/dense_138/Tensordot/stackPack>multi_head_self_attention_15/dense_138/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_138/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_138/Tensordot/stack?
:multi_head_self_attention_15/dense_138/Tensordot/transpose	Transpose/multi_head_self_attention_15/Reshape_3:output:0@multi_head_self_attention_15/dense_138/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2<
:multi_head_self_attention_15/dense_138/Tensordot/transpose?
8multi_head_self_attention_15/dense_138/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_138/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_138/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_138/Tensordot/Reshape?
7multi_head_self_attention_15/dense_138/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_138/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_138/Tensordot/MatMul?
8multi_head_self_attention_15/dense_138/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_138/Tensordot/Const_2?
>multi_head_self_attention_15/dense_138/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_138/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_138/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_138/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_138/Tensordot/concat_1?
0multi_head_self_attention_15/dense_138/TensordotReshapeAmulti_head_self_attention_15/dense_138/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_138/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 22
0multi_head_self_attention_15/dense_138/Tensordot?
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_138/BiasAddBiasAdd9multi_head_self_attention_15/dense_138/Tensordot:output:0Emulti_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_15/dense_138/BiasAdd?
dropout_30/IdentityIdentity7multi_head_self_attention_15/dense_138/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_30/Identityo
addAddV2inputsdropout_30/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add?
5layer_normalization_30/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_30/moments/mean/reduction_indices?
#layer_normalization_30/moments/meanMeanadd:z:0>layer_normalization_30/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_30/moments/mean?
+layer_normalization_30/moments/StopGradientStopGradient,layer_normalization_30/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_30/moments/StopGradient?
0layer_normalization_30/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_30/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_30/moments/SquaredDifference?
9layer_normalization_30/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_30/moments/variance/reduction_indices?
'layer_normalization_30/moments/varianceMean4layer_normalization_30/moments/SquaredDifference:z:0Blayer_normalization_30/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_30/moments/variance?
&layer_normalization_30/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_30/batchnorm/add/y?
$layer_normalization_30/batchnorm/addAddV20layer_normalization_30/moments/variance:output:0/layer_normalization_30/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_30/batchnorm/add?
&layer_normalization_30/batchnorm/RsqrtRsqrt(layer_normalization_30/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_30/batchnorm/Rsqrt?
3layer_normalization_30/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_30_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_30/batchnorm/mul/ReadVariableOp?
$layer_normalization_30/batchnorm/mulMul*layer_normalization_30/batchnorm/Rsqrt:y:0;layer_normalization_30/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_30/batchnorm/mul?
&layer_normalization_30/batchnorm/mul_1Muladd:z:0(layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/mul_1?
&layer_normalization_30/batchnorm/mul_2Mul,layer_normalization_30/moments/mean:output:0(layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/mul_2?
/layer_normalization_30/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_30_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_30/batchnorm/ReadVariableOp?
$layer_normalization_30/batchnorm/subSub7layer_normalization_30/batchnorm/ReadVariableOp:value:0*layer_normalization_30/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_30/batchnorm/sub?
&layer_normalization_30/batchnorm/add_1AddV2*layer_normalization_30/batchnorm/mul_1:z:0(layer_normalization_30/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/add_1?
0sequential_15/dense_139/Tensordot/ReadVariableOpReadVariableOp9sequential_15_dense_139_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype022
0sequential_15/dense_139/Tensordot/ReadVariableOp?
&sequential_15/dense_139/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_15/dense_139/Tensordot/axes?
&sequential_15/dense_139/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&sequential_15/dense_139/Tensordot/free?
'sequential_15/dense_139/Tensordot/ShapeShape*layer_normalization_30/batchnorm/add_1:z:0*
T0*
_output_shapes
:2)
'sequential_15/dense_139/Tensordot/Shape?
/sequential_15/dense_139/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_139/Tensordot/GatherV2/axis?
*sequential_15/dense_139/Tensordot/GatherV2GatherV20sequential_15/dense_139/Tensordot/Shape:output:0/sequential_15/dense_139/Tensordot/free:output:08sequential_15/dense_139/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_15/dense_139/Tensordot/GatherV2?
1sequential_15/dense_139/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_15/dense_139/Tensordot/GatherV2_1/axis?
,sequential_15/dense_139/Tensordot/GatherV2_1GatherV20sequential_15/dense_139/Tensordot/Shape:output:0/sequential_15/dense_139/Tensordot/axes:output:0:sequential_15/dense_139/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,sequential_15/dense_139/Tensordot/GatherV2_1?
'sequential_15/dense_139/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_15/dense_139/Tensordot/Const?
&sequential_15/dense_139/Tensordot/ProdProd3sequential_15/dense_139/Tensordot/GatherV2:output:00sequential_15/dense_139/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&sequential_15/dense_139/Tensordot/Prod?
)sequential_15/dense_139/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_139/Tensordot/Const_1?
(sequential_15/dense_139/Tensordot/Prod_1Prod5sequential_15/dense_139/Tensordot/GatherV2_1:output:02sequential_15/dense_139/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(sequential_15/dense_139/Tensordot/Prod_1?
-sequential_15/dense_139/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_15/dense_139/Tensordot/concat/axis?
(sequential_15/dense_139/Tensordot/concatConcatV2/sequential_15/dense_139/Tensordot/free:output:0/sequential_15/dense_139/Tensordot/axes:output:06sequential_15/dense_139/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_15/dense_139/Tensordot/concat?
'sequential_15/dense_139/Tensordot/stackPack/sequential_15/dense_139/Tensordot/Prod:output:01sequential_15/dense_139/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_15/dense_139/Tensordot/stack?
+sequential_15/dense_139/Tensordot/transpose	Transpose*layer_normalization_30/batchnorm/add_1:z:01sequential_15/dense_139/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2-
+sequential_15/dense_139/Tensordot/transpose?
)sequential_15/dense_139/Tensordot/ReshapeReshape/sequential_15/dense_139/Tensordot/transpose:y:00sequential_15/dense_139/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)sequential_15/dense_139/Tensordot/Reshape?
(sequential_15/dense_139/Tensordot/MatMulMatMul2sequential_15/dense_139/Tensordot/Reshape:output:08sequential_15/dense_139/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(sequential_15/dense_139/Tensordot/MatMul?
)sequential_15/dense_139/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_139/Tensordot/Const_2?
/sequential_15/dense_139/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_139/Tensordot/concat_1/axis?
*sequential_15/dense_139/Tensordot/concat_1ConcatV23sequential_15/dense_139/Tensordot/GatherV2:output:02sequential_15/dense_139/Tensordot/Const_2:output:08sequential_15/dense_139/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*sequential_15/dense_139/Tensordot/concat_1?
!sequential_15/dense_139/TensordotReshape2sequential_15/dense_139/Tensordot/MatMul:product:03sequential_15/dense_139/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2#
!sequential_15/dense_139/Tensordot?
.sequential_15/dense_139/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_dense_139_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/dense_139/BiasAdd/ReadVariableOp?
sequential_15/dense_139/BiasAddBiasAdd*sequential_15/dense_139/Tensordot:output:06sequential_15/dense_139/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2!
sequential_15/dense_139/BiasAdd?
sequential_15/dense_139/ReluRelu(sequential_15/dense_139/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_15/dense_139/Relu?
0sequential_15/dense_140/Tensordot/ReadVariableOpReadVariableOp9sequential_15_dense_140_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype022
0sequential_15/dense_140/Tensordot/ReadVariableOp?
&sequential_15/dense_140/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_15/dense_140/Tensordot/axes?
&sequential_15/dense_140/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&sequential_15/dense_140/Tensordot/free?
'sequential_15/dense_140/Tensordot/ShapeShape*sequential_15/dense_139/Relu:activations:0*
T0*
_output_shapes
:2)
'sequential_15/dense_140/Tensordot/Shape?
/sequential_15/dense_140/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_140/Tensordot/GatherV2/axis?
*sequential_15/dense_140/Tensordot/GatherV2GatherV20sequential_15/dense_140/Tensordot/Shape:output:0/sequential_15/dense_140/Tensordot/free:output:08sequential_15/dense_140/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_15/dense_140/Tensordot/GatherV2?
1sequential_15/dense_140/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_15/dense_140/Tensordot/GatherV2_1/axis?
,sequential_15/dense_140/Tensordot/GatherV2_1GatherV20sequential_15/dense_140/Tensordot/Shape:output:0/sequential_15/dense_140/Tensordot/axes:output:0:sequential_15/dense_140/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,sequential_15/dense_140/Tensordot/GatherV2_1?
'sequential_15/dense_140/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_15/dense_140/Tensordot/Const?
&sequential_15/dense_140/Tensordot/ProdProd3sequential_15/dense_140/Tensordot/GatherV2:output:00sequential_15/dense_140/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&sequential_15/dense_140/Tensordot/Prod?
)sequential_15/dense_140/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_140/Tensordot/Const_1?
(sequential_15/dense_140/Tensordot/Prod_1Prod5sequential_15/dense_140/Tensordot/GatherV2_1:output:02sequential_15/dense_140/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(sequential_15/dense_140/Tensordot/Prod_1?
-sequential_15/dense_140/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_15/dense_140/Tensordot/concat/axis?
(sequential_15/dense_140/Tensordot/concatConcatV2/sequential_15/dense_140/Tensordot/free:output:0/sequential_15/dense_140/Tensordot/axes:output:06sequential_15/dense_140/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_15/dense_140/Tensordot/concat?
'sequential_15/dense_140/Tensordot/stackPack/sequential_15/dense_140/Tensordot/Prod:output:01sequential_15/dense_140/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_15/dense_140/Tensordot/stack?
+sequential_15/dense_140/Tensordot/transpose	Transpose*sequential_15/dense_139/Relu:activations:01sequential_15/dense_140/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2-
+sequential_15/dense_140/Tensordot/transpose?
)sequential_15/dense_140/Tensordot/ReshapeReshape/sequential_15/dense_140/Tensordot/transpose:y:00sequential_15/dense_140/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)sequential_15/dense_140/Tensordot/Reshape?
(sequential_15/dense_140/Tensordot/MatMulMatMul2sequential_15/dense_140/Tensordot/Reshape:output:08sequential_15/dense_140/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(sequential_15/dense_140/Tensordot/MatMul?
)sequential_15/dense_140/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_140/Tensordot/Const_2?
/sequential_15/dense_140/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_140/Tensordot/concat_1/axis?
*sequential_15/dense_140/Tensordot/concat_1ConcatV23sequential_15/dense_140/Tensordot/GatherV2:output:02sequential_15/dense_140/Tensordot/Const_2:output:08sequential_15/dense_140/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*sequential_15/dense_140/Tensordot/concat_1?
!sequential_15/dense_140/TensordotReshape2sequential_15/dense_140/Tensordot/MatMul:product:03sequential_15/dense_140/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2#
!sequential_15/dense_140/Tensordot?
.sequential_15/dense_140/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/dense_140/BiasAdd/ReadVariableOp?
sequential_15/dense_140/BiasAddBiasAdd*sequential_15/dense_140/Tensordot:output:06sequential_15/dense_140/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2!
sequential_15/dense_140/BiasAdd?
dropout_31/IdentityIdentity(sequential_15/dense_140/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_31/Identity?
add_1AddV2*layer_normalization_30/batchnorm/add_1:z:0dropout_31/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add_1?
5layer_normalization_31/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_31/moments/mean/reduction_indices?
#layer_normalization_31/moments/meanMean	add_1:z:0>layer_normalization_31/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_31/moments/mean?
+layer_normalization_31/moments/StopGradientStopGradient,layer_normalization_31/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_31/moments/StopGradient?
0layer_normalization_31/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_31/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_31/moments/SquaredDifference?
9layer_normalization_31/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_31/moments/variance/reduction_indices?
'layer_normalization_31/moments/varianceMean4layer_normalization_31/moments/SquaredDifference:z:0Blayer_normalization_31/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_31/moments/variance?
&layer_normalization_31/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_31/batchnorm/add/y?
$layer_normalization_31/batchnorm/addAddV20layer_normalization_31/moments/variance:output:0/layer_normalization_31/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_31/batchnorm/add?
&layer_normalization_31/batchnorm/RsqrtRsqrt(layer_normalization_31/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_31/batchnorm/Rsqrt?
3layer_normalization_31/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_31_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_31/batchnorm/mul/ReadVariableOp?
$layer_normalization_31/batchnorm/mulMul*layer_normalization_31/batchnorm/Rsqrt:y:0;layer_normalization_31/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_31/batchnorm/mul?
&layer_normalization_31/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/mul_1?
&layer_normalization_31/batchnorm/mul_2Mul,layer_normalization_31/moments/mean:output:0(layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/mul_2?
/layer_normalization_31/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_31_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_31/batchnorm/ReadVariableOp?
$layer_normalization_31/batchnorm/subSub7layer_normalization_31/batchnorm/ReadVariableOp:value:0*layer_normalization_31/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_31/batchnorm/sub?
&layer_normalization_31/batchnorm/add_1AddV2*layer_normalization_31/batchnorm/mul_1:z:0(layer_normalization_31/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/add_1?
IdentityIdentity*layer_normalization_31/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp0^layer_normalization_30/batchnorm/ReadVariableOp4^layer_normalization_30/batchnorm/mul/ReadVariableOp0^layer_normalization_31/batchnorm/ReadVariableOp4^layer_normalization_31/batchnorm/mul/ReadVariableOp>^multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp/^sequential_15/dense_139/BiasAdd/ReadVariableOp1^sequential_15/dense_139/Tensordot/ReadVariableOp/^sequential_15/dense_140/BiasAdd/ReadVariableOp1^sequential_15/dense_140/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2b
/layer_normalization_30/batchnorm/ReadVariableOp/layer_normalization_30/batchnorm/ReadVariableOp2j
3layer_normalization_30/batchnorm/mul/ReadVariableOp3layer_normalization_30/batchnorm/mul/ReadVariableOp2b
/layer_normalization_31/batchnorm/ReadVariableOp/layer_normalization_31/batchnorm/ReadVariableOp2j
3layer_normalization_31/batchnorm/mul/ReadVariableOp3layer_normalization_31/batchnorm/mul/ReadVariableOp2~
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp2`
.sequential_15/dense_139/BiasAdd/ReadVariableOp.sequential_15/dense_139/BiasAdd/ReadVariableOp2d
0sequential_15/dense_139/Tensordot/ReadVariableOp0sequential_15/dense_139/Tensordot/ReadVariableOp2`
.sequential_15/dense_140/BiasAdd/ReadVariableOp.sequential_15/dense_140/BiasAdd/ReadVariableOp2d
0sequential_15/dense_140/Tensordot/ReadVariableOp0sequential_15/dense_140/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
F__inference_dense_141_layer_call_and_return_conditional_losses_6977987

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_concatenate_15_layer_call_and_return_conditional_losses_6975531

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_142_layer_call_fn_6978016

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_142_layer_call_and_return_conditional_losses_69755612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
־
?J
#__inference__traced_restore_6978905
file_prefix4
"assignvariableop_aux_output_kernel: 0
"assignvariableop_1_aux_output_bias:5
#assignvariableop_2_dense_141_kernel:@/
!assignvariableop_3_dense_141_bias:@5
#assignvariableop_4_dense_142_kernel:@@/
!assignvariableop_5_dense_142_bias:@5
#assignvariableop_6_dense_143_kernel:@@/
!assignvariableop_7_dense_143_bias:@7
%assignvariableop_8_main_output_kernel:@1
#assignvariableop_9_main_output_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: ]
Kassignvariableop_15_token_and_position_embedding_15_embedding_30_embeddings: ]
Kassignvariableop_16_token_and_position_embedding_15_embedding_31_embeddings:( h
Vassignvariableop_17_transformer_block_15_multi_head_self_attention_15_dense_135_kernel:  b
Tassignvariableop_18_transformer_block_15_multi_head_self_attention_15_dense_135_bias: h
Vassignvariableop_19_transformer_block_15_multi_head_self_attention_15_dense_136_kernel:  b
Tassignvariableop_20_transformer_block_15_multi_head_self_attention_15_dense_136_bias: h
Vassignvariableop_21_transformer_block_15_multi_head_self_attention_15_dense_137_kernel:  b
Tassignvariableop_22_transformer_block_15_multi_head_self_attention_15_dense_137_bias: h
Vassignvariableop_23_transformer_block_15_multi_head_self_attention_15_dense_138_kernel:  b
Tassignvariableop_24_transformer_block_15_multi_head_self_attention_15_dense_138_bias: 6
$assignvariableop_25_dense_139_kernel:  0
"assignvariableop_26_dense_139_bias: 6
$assignvariableop_27_dense_140_kernel:  0
"assignvariableop_28_dense_140_bias: S
Eassignvariableop_29_transformer_block_15_layer_normalization_30_gamma: R
Dassignvariableop_30_transformer_block_15_layer_normalization_30_beta: S
Eassignvariableop_31_transformer_block_15_layer_normalization_31_gamma: R
Dassignvariableop_32_transformer_block_15_layer_normalization_31_beta: #
assignvariableop_33_total: #
assignvariableop_34_count: %
assignvariableop_35_total_1: %
assignvariableop_36_count_1: %
assignvariableop_37_total_2: %
assignvariableop_38_count_2: %
assignvariableop_39_total_3: %
assignvariableop_40_count_3: %
assignvariableop_41_total_4: %
assignvariableop_42_count_4: >
,assignvariableop_43_adam_aux_output_kernel_m: 8
*assignvariableop_44_adam_aux_output_bias_m:=
+assignvariableop_45_adam_dense_141_kernel_m:@7
)assignvariableop_46_adam_dense_141_bias_m:@=
+assignvariableop_47_adam_dense_142_kernel_m:@@7
)assignvariableop_48_adam_dense_142_bias_m:@=
+assignvariableop_49_adam_dense_143_kernel_m:@@7
)assignvariableop_50_adam_dense_143_bias_m:@?
-assignvariableop_51_adam_main_output_kernel_m:@9
+assignvariableop_52_adam_main_output_bias_m:d
Rassignvariableop_53_adam_token_and_position_embedding_15_embedding_30_embeddings_m: d
Rassignvariableop_54_adam_token_and_position_embedding_15_embedding_31_embeddings_m:( o
]assignvariableop_55_adam_transformer_block_15_multi_head_self_attention_15_dense_135_kernel_m:  i
[assignvariableop_56_adam_transformer_block_15_multi_head_self_attention_15_dense_135_bias_m: o
]assignvariableop_57_adam_transformer_block_15_multi_head_self_attention_15_dense_136_kernel_m:  i
[assignvariableop_58_adam_transformer_block_15_multi_head_self_attention_15_dense_136_bias_m: o
]assignvariableop_59_adam_transformer_block_15_multi_head_self_attention_15_dense_137_kernel_m:  i
[assignvariableop_60_adam_transformer_block_15_multi_head_self_attention_15_dense_137_bias_m: o
]assignvariableop_61_adam_transformer_block_15_multi_head_self_attention_15_dense_138_kernel_m:  i
[assignvariableop_62_adam_transformer_block_15_multi_head_self_attention_15_dense_138_bias_m: =
+assignvariableop_63_adam_dense_139_kernel_m:  7
)assignvariableop_64_adam_dense_139_bias_m: =
+assignvariableop_65_adam_dense_140_kernel_m:  7
)assignvariableop_66_adam_dense_140_bias_m: Z
Lassignvariableop_67_adam_transformer_block_15_layer_normalization_30_gamma_m: Y
Kassignvariableop_68_adam_transformer_block_15_layer_normalization_30_beta_m: Z
Lassignvariableop_69_adam_transformer_block_15_layer_normalization_31_gamma_m: Y
Kassignvariableop_70_adam_transformer_block_15_layer_normalization_31_beta_m: >
,assignvariableop_71_adam_aux_output_kernel_v: 8
*assignvariableop_72_adam_aux_output_bias_v:=
+assignvariableop_73_adam_dense_141_kernel_v:@7
)assignvariableop_74_adam_dense_141_bias_v:@=
+assignvariableop_75_adam_dense_142_kernel_v:@@7
)assignvariableop_76_adam_dense_142_bias_v:@=
+assignvariableop_77_adam_dense_143_kernel_v:@@7
)assignvariableop_78_adam_dense_143_bias_v:@?
-assignvariableop_79_adam_main_output_kernel_v:@9
+assignvariableop_80_adam_main_output_bias_v:d
Rassignvariableop_81_adam_token_and_position_embedding_15_embedding_30_embeddings_v: d
Rassignvariableop_82_adam_token_and_position_embedding_15_embedding_31_embeddings_v:( o
]assignvariableop_83_adam_transformer_block_15_multi_head_self_attention_15_dense_135_kernel_v:  i
[assignvariableop_84_adam_transformer_block_15_multi_head_self_attention_15_dense_135_bias_v: o
]assignvariableop_85_adam_transformer_block_15_multi_head_self_attention_15_dense_136_kernel_v:  i
[assignvariableop_86_adam_transformer_block_15_multi_head_self_attention_15_dense_136_bias_v: o
]assignvariableop_87_adam_transformer_block_15_multi_head_self_attention_15_dense_137_kernel_v:  i
[assignvariableop_88_adam_transformer_block_15_multi_head_self_attention_15_dense_137_bias_v: o
]assignvariableop_89_adam_transformer_block_15_multi_head_self_attention_15_dense_138_kernel_v:  i
[assignvariableop_90_adam_transformer_block_15_multi_head_self_attention_15_dense_138_bias_v: =
+assignvariableop_91_adam_dense_139_kernel_v:  7
)assignvariableop_92_adam_dense_139_bias_v: =
+assignvariableop_93_adam_dense_140_kernel_v:  7
)assignvariableop_94_adam_dense_140_bias_v: Z
Lassignvariableop_95_adam_transformer_block_15_layer_normalization_30_gamma_v: Y
Kassignvariableop_96_adam_transformer_block_15_layer_normalization_30_beta_v: Z
Lassignvariableop_97_adam_transformer_block_15_layer_normalization_31_gamma_v: Y
Kassignvariableop_98_adam_transformer_block_15_layer_normalization_31_beta_v: 
identity_100??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?5
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?4
value?4B?4dB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?
value?B?dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*r
dtypesh
f2d	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_aux_output_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_aux_output_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_141_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_141_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_142_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_142_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_143_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_143_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_main_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_main_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpKassignvariableop_15_token_and_position_embedding_15_embedding_30_embeddingsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpKassignvariableop_16_token_and_position_embedding_15_embedding_31_embeddingsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpVassignvariableop_17_transformer_block_15_multi_head_self_attention_15_dense_135_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpTassignvariableop_18_transformer_block_15_multi_head_self_attention_15_dense_135_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpVassignvariableop_19_transformer_block_15_multi_head_self_attention_15_dense_136_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpTassignvariableop_20_transformer_block_15_multi_head_self_attention_15_dense_136_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpVassignvariableop_21_transformer_block_15_multi_head_self_attention_15_dense_137_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpTassignvariableop_22_transformer_block_15_multi_head_self_attention_15_dense_137_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpVassignvariableop_23_transformer_block_15_multi_head_self_attention_15_dense_138_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpTassignvariableop_24_transformer_block_15_multi_head_self_attention_15_dense_138_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp$assignvariableop_25_dense_139_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_139_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_140_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_140_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpEassignvariableop_29_transformer_block_15_layer_normalization_30_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpDassignvariableop_30_transformer_block_15_layer_normalization_30_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpEassignvariableop_31_transformer_block_15_layer_normalization_31_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpDassignvariableop_32_transformer_block_15_layer_normalization_31_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_total_2Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_count_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_3Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_3Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_4Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_4Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_aux_output_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_aux_output_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_dense_141_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_141_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_142_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_142_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_dense_143_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_143_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_main_output_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp+assignvariableop_52_adam_main_output_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpRassignvariableop_53_adam_token_and_position_embedding_15_embedding_30_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpRassignvariableop_54_adam_token_and_position_embedding_15_embedding_31_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp]assignvariableop_55_adam_transformer_block_15_multi_head_self_attention_15_dense_135_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp[assignvariableop_56_adam_transformer_block_15_multi_head_self_attention_15_dense_135_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp]assignvariableop_57_adam_transformer_block_15_multi_head_self_attention_15_dense_136_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp[assignvariableop_58_adam_transformer_block_15_multi_head_self_attention_15_dense_136_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp]assignvariableop_59_adam_transformer_block_15_multi_head_self_attention_15_dense_137_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp[assignvariableop_60_adam_transformer_block_15_multi_head_self_attention_15_dense_137_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp]assignvariableop_61_adam_transformer_block_15_multi_head_self_attention_15_dense_138_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp[assignvariableop_62_adam_transformer_block_15_multi_head_self_attention_15_dense_138_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_dense_139_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_dense_139_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_dense_140_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_dense_140_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpLassignvariableop_67_adam_transformer_block_15_layer_normalization_30_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpKassignvariableop_68_adam_transformer_block_15_layer_normalization_30_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpLassignvariableop_69_adam_transformer_block_15_layer_normalization_31_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOpKassignvariableop_70_adam_transformer_block_15_layer_normalization_31_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_aux_output_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_aux_output_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_dense_141_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_dense_141_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_dense_142_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_dense_142_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_dense_143_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_dense_143_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp-assignvariableop_79_adam_main_output_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp+assignvariableop_80_adam_main_output_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOpRassignvariableop_81_adam_token_and_position_embedding_15_embedding_30_embeddings_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOpRassignvariableop_82_adam_token_and_position_embedding_15_embedding_31_embeddings_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp]assignvariableop_83_adam_transformer_block_15_multi_head_self_attention_15_dense_135_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp[assignvariableop_84_adam_transformer_block_15_multi_head_self_attention_15_dense_135_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp]assignvariableop_85_adam_transformer_block_15_multi_head_self_attention_15_dense_136_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp[assignvariableop_86_adam_transformer_block_15_multi_head_self_attention_15_dense_136_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp]assignvariableop_87_adam_transformer_block_15_multi_head_self_attention_15_dense_137_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp[assignvariableop_88_adam_transformer_block_15_multi_head_self_attention_15_dense_137_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp]assignvariableop_89_adam_transformer_block_15_multi_head_self_attention_15_dense_138_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp[assignvariableop_90_adam_transformer_block_15_multi_head_self_attention_15_dense_138_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_dense_139_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_dense_139_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_dense_140_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_dense_140_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOpLassignvariableop_95_adam_transformer_block_15_layer_normalization_30_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOpKassignvariableop_96_adam_transformer_block_15_layer_normalization_30_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOpLassignvariableop_97_adam_transformer_block_15_layer_normalization_31_gamma_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOpKassignvariableop_98_adam_transformer_block_15_layer_normalization_31_beta_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_989
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_99Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_99h
Identity_100IdentityIdentity_99:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_100?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_100Identity_100:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_98:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
\__inference_token_and_position_embedding_15_layer_call_and_return_conditional_losses_6975215
x7
%embedding_31_embedding_lookup_6975202:( 7
%embedding_30_embedding_lookup_6975208: 
identity??embedding_30/embedding_lookup?embedding_31/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta?
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:?????????2
range?
embedding_31/embedding_lookupResourceGather%embedding_31_embedding_lookup_6975202range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_31/embedding_lookup/6975202*'
_output_shapes
:????????? *
dtype02
embedding_31/embedding_lookup?
&embedding_31/embedding_lookup/IdentityIdentity&embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_31/embedding_lookup/6975202*'
_output_shapes
:????????? 2(
&embedding_31/embedding_lookup/Identity?
(embedding_31/embedding_lookup/Identity_1Identity/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2*
(embedding_31/embedding_lookup/Identity_1r
embedding_30/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:?????????(2
embedding_30/Cast?
embedding_30/embedding_lookupResourceGather%embedding_30_embedding_lookup_6975208embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_30/embedding_lookup/6975208*+
_output_shapes
:?????????( *
dtype02
embedding_30/embedding_lookup?
&embedding_30/embedding_lookup/IdentityIdentity&embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_30/embedding_lookup/6975208*+
_output_shapes
:?????????( 2(
&embedding_30/embedding_lookup/Identity?
(embedding_30/embedding_lookup/Identity_1Identity/embedding_30/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2*
(embedding_30/embedding_lookup/Identity_1?
addAddV21embedding_30/embedding_lookup/Identity_1:output:01embedding_31/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2
addf
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp^embedding_30/embedding_lookup^embedding_31/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 2>
embedding_30/embedding_lookupembedding_30/embedding_lookup2>
embedding_31/embedding_lookupembedding_31/embedding_lookup:J F
'
_output_shapes
:?????????(

_user_specified_namex
??
?
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_6977845

inputsZ
Hmulti_head_self_attention_15_dense_135_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_135_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_136_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_136_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_137_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_137_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_138_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_138_biasadd_readvariableop_resource: J
<layer_normalization_30_batchnorm_mul_readvariableop_resource: F
8layer_normalization_30_batchnorm_readvariableop_resource: K
9sequential_15_dense_139_tensordot_readvariableop_resource:  E
7sequential_15_dense_139_biasadd_readvariableop_resource: K
9sequential_15_dense_140_tensordot_readvariableop_resource:  E
7sequential_15_dense_140_biasadd_readvariableop_resource: J
<layer_normalization_31_batchnorm_mul_readvariableop_resource: F
8layer_normalization_31_batchnorm_readvariableop_resource: 
identity??/layer_normalization_30/batchnorm/ReadVariableOp?3layer_normalization_30/batchnorm/mul/ReadVariableOp?/layer_normalization_31/batchnorm/ReadVariableOp?3layer_normalization_31/batchnorm/mul/ReadVariableOp?=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?.sequential_15/dense_139/BiasAdd/ReadVariableOp?0sequential_15/dense_139/Tensordot/ReadVariableOp?.sequential_15/dense_140/BiasAdd/ReadVariableOp?0sequential_15/dense_140/Tensordot/ReadVariableOp~
"multi_head_self_attention_15/ShapeShapeinputs*
T0*
_output_shapes
:2$
"multi_head_self_attention_15/Shape?
0multi_head_self_attention_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0multi_head_self_attention_15/strided_slice/stack?
2multi_head_self_attention_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2multi_head_self_attention_15/strided_slice/stack_1?
2multi_head_self_attention_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2multi_head_self_attention_15/strided_slice/stack_2?
*multi_head_self_attention_15/strided_sliceStridedSlice+multi_head_self_attention_15/Shape:output:09multi_head_self_attention_15/strided_slice/stack:output:0;multi_head_self_attention_15/strided_slice/stack_1:output:0;multi_head_self_attention_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*multi_head_self_attention_15/strided_slice?
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_135_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_135/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_135/Tensordot/axes?
5multi_head_self_attention_15/dense_135/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_135/Tensordot/free?
6multi_head_self_attention_15/dense_135/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_135/Tensordot/Shape?
>multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_135/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_135/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_135/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_135/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_135/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_135/Tensordot/Const?
5multi_head_self_attention_15/dense_135/Tensordot/ProdProdBmulti_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_135/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_135/Tensordot/Prod?
8multi_head_self_attention_15/dense_135/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_135/Tensordot/Const_1?
7multi_head_self_attention_15/dense_135/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_135/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_135/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_135/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_135/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_135/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_135/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_135/Tensordot/free:output:0>multi_head_self_attention_15/dense_135/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_135/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_135/Tensordot/concat?
6multi_head_self_attention_15/dense_135/Tensordot/stackPack>multi_head_self_attention_15/dense_135/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_135/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_135/Tensordot/stack?
:multi_head_self_attention_15/dense_135/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_135/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_135/Tensordot/transpose?
8multi_head_self_attention_15/dense_135/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_135/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_135/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_135/Tensordot/Reshape?
7multi_head_self_attention_15/dense_135/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_135/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_135/Tensordot/MatMul?
8multi_head_self_attention_15/dense_135/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_135/Tensordot/Const_2?
>multi_head_self_attention_15/dense_135/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_135/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_135/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_135/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_135/Tensordot/concat_1?
0multi_head_self_attention_15/dense_135/TensordotReshapeAmulti_head_self_attention_15/dense_135/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_135/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_135/Tensordot?
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_135_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_135/BiasAddBiasAdd9multi_head_self_attention_15/dense_135/Tensordot:output:0Emulti_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_135/BiasAdd?
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_136_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_136/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_136/Tensordot/axes?
5multi_head_self_attention_15/dense_136/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_136/Tensordot/free?
6multi_head_self_attention_15/dense_136/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_136/Tensordot/Shape?
>multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_136/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_136/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_136/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_136/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_136/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_136/Tensordot/Const?
5multi_head_self_attention_15/dense_136/Tensordot/ProdProdBmulti_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_136/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_136/Tensordot/Prod?
8multi_head_self_attention_15/dense_136/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_136/Tensordot/Const_1?
7multi_head_self_attention_15/dense_136/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_136/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_136/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_136/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_136/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_136/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_136/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_136/Tensordot/free:output:0>multi_head_self_attention_15/dense_136/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_136/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_136/Tensordot/concat?
6multi_head_self_attention_15/dense_136/Tensordot/stackPack>multi_head_self_attention_15/dense_136/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_136/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_136/Tensordot/stack?
:multi_head_self_attention_15/dense_136/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_136/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_136/Tensordot/transpose?
8multi_head_self_attention_15/dense_136/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_136/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_136/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_136/Tensordot/Reshape?
7multi_head_self_attention_15/dense_136/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_136/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_136/Tensordot/MatMul?
8multi_head_self_attention_15/dense_136/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_136/Tensordot/Const_2?
>multi_head_self_attention_15/dense_136/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_136/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_136/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_136/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_136/Tensordot/concat_1?
0multi_head_self_attention_15/dense_136/TensordotReshapeAmulti_head_self_attention_15/dense_136/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_136/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_136/Tensordot?
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_136_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_136/BiasAddBiasAdd9multi_head_self_attention_15/dense_136/Tensordot:output:0Emulti_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_136/BiasAdd?
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_137_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_137/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_137/Tensordot/axes?
5multi_head_self_attention_15/dense_137/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_137/Tensordot/free?
6multi_head_self_attention_15/dense_137/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_137/Tensordot/Shape?
>multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_137/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_137/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_137/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_137/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_137/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_137/Tensordot/Const?
5multi_head_self_attention_15/dense_137/Tensordot/ProdProdBmulti_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_137/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_137/Tensordot/Prod?
8multi_head_self_attention_15/dense_137/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_137/Tensordot/Const_1?
7multi_head_self_attention_15/dense_137/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_137/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_137/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_137/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_137/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_137/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_137/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_137/Tensordot/free:output:0>multi_head_self_attention_15/dense_137/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_137/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_137/Tensordot/concat?
6multi_head_self_attention_15/dense_137/Tensordot/stackPack>multi_head_self_attention_15/dense_137/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_137/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_137/Tensordot/stack?
:multi_head_self_attention_15/dense_137/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_137/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_137/Tensordot/transpose?
8multi_head_self_attention_15/dense_137/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_137/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_137/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_137/Tensordot/Reshape?
7multi_head_self_attention_15/dense_137/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_137/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_137/Tensordot/MatMul?
8multi_head_self_attention_15/dense_137/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_137/Tensordot/Const_2?
>multi_head_self_attention_15/dense_137/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_137/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_137/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_137/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_137/Tensordot/concat_1?
0multi_head_self_attention_15/dense_137/TensordotReshapeAmulti_head_self_attention_15/dense_137/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_137/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_137/Tensordot?
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_137_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_137/BiasAddBiasAdd9multi_head_self_attention_15/dense_137/Tensordot:output:0Emulti_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_137/BiasAdd?
,multi_head_self_attention_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,multi_head_self_attention_15/Reshape/shape/1?
,multi_head_self_attention_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2.
,multi_head_self_attention_15/Reshape/shape/2?
,multi_head_self_attention_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2.
,multi_head_self_attention_15/Reshape/shape/3?
*multi_head_self_attention_15/Reshape/shapePack3multi_head_self_attention_15/strided_slice:output:05multi_head_self_attention_15/Reshape/shape/1:output:05multi_head_self_attention_15/Reshape/shape/2:output:05multi_head_self_attention_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2,
*multi_head_self_attention_15/Reshape/shape?
$multi_head_self_attention_15/ReshapeReshape7multi_head_self_attention_15/dense_135/BiasAdd:output:03multi_head_self_attention_15/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_15/Reshape?
+multi_head_self_attention_15/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+multi_head_self_attention_15/transpose/perm?
&multi_head_self_attention_15/transpose	Transpose-multi_head_self_attention_15/Reshape:output:04multi_head_self_attention_15/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/transpose?
.multi_head_self_attention_15/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_1/shape/1?
.multi_head_self_attention_15/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_1/shape/2?
.multi_head_self_attention_15/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_1/shape/3?
,multi_head_self_attention_15/Reshape_1/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_1/shape/1:output:07multi_head_self_attention_15/Reshape_1/shape/2:output:07multi_head_self_attention_15/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_1/shape?
&multi_head_self_attention_15/Reshape_1Reshape7multi_head_self_attention_15/dense_136/BiasAdd:output:05multi_head_self_attention_15/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/Reshape_1?
-multi_head_self_attention_15/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_1/perm?
(multi_head_self_attention_15/transpose_1	Transpose/multi_head_self_attention_15/Reshape_1:output:06multi_head_self_attention_15/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_1?
.multi_head_self_attention_15/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_2/shape/1?
.multi_head_self_attention_15/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_2/shape/2?
.multi_head_self_attention_15/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_2/shape/3?
,multi_head_self_attention_15/Reshape_2/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_2/shape/1:output:07multi_head_self_attention_15/Reshape_2/shape/2:output:07multi_head_self_attention_15/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_2/shape?
&multi_head_self_attention_15/Reshape_2Reshape7multi_head_self_attention_15/dense_137/BiasAdd:output:05multi_head_self_attention_15/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/Reshape_2?
-multi_head_self_attention_15/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_2/perm?
(multi_head_self_attention_15/transpose_2	Transpose/multi_head_self_attention_15/Reshape_2:output:06multi_head_self_attention_15/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_2?
#multi_head_self_attention_15/MatMulBatchMatMulV2*multi_head_self_attention_15/transpose:y:0,multi_head_self_attention_15/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2%
#multi_head_self_attention_15/MatMul?
$multi_head_self_attention_15/Shape_1Shape,multi_head_self_attention_15/transpose_1:y:0*
T0*
_output_shapes
:2&
$multi_head_self_attention_15/Shape_1?
2multi_head_self_attention_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2multi_head_self_attention_15/strided_slice_1/stack?
4multi_head_self_attention_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_15/strided_slice_1/stack_1?
4multi_head_self_attention_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4multi_head_self_attention_15/strided_slice_1/stack_2?
,multi_head_self_attention_15/strided_slice_1StridedSlice-multi_head_self_attention_15/Shape_1:output:0;multi_head_self_attention_15/strided_slice_1/stack:output:0=multi_head_self_attention_15/strided_slice_1/stack_1:output:0=multi_head_self_attention_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,multi_head_self_attention_15/strided_slice_1?
!multi_head_self_attention_15/CastCast5multi_head_self_attention_15/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!multi_head_self_attention_15/Cast?
!multi_head_self_attention_15/SqrtSqrt%multi_head_self_attention_15/Cast:y:0*
T0*
_output_shapes
: 2#
!multi_head_self_attention_15/Sqrt?
$multi_head_self_attention_15/truedivRealDiv,multi_head_self_attention_15/MatMul:output:0%multi_head_self_attention_15/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2&
$multi_head_self_attention_15/truediv?
$multi_head_self_attention_15/SoftmaxSoftmax(multi_head_self_attention_15/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2&
$multi_head_self_attention_15/Softmax?
%multi_head_self_attention_15/MatMul_1BatchMatMulV2.multi_head_self_attention_15/Softmax:softmax:0,multi_head_self_attention_15/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_15/MatMul_1?
-multi_head_self_attention_15/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_3/perm?
(multi_head_self_attention_15/transpose_3	Transpose.multi_head_self_attention_15/MatMul_1:output:06multi_head_self_attention_15/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_3?
.multi_head_self_attention_15/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_3/shape/1?
.multi_head_self_attention_15/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_self_attention_15/Reshape_3/shape/2?
,multi_head_self_attention_15/Reshape_3/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_3/shape/1:output:07multi_head_self_attention_15/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_3/shape?
&multi_head_self_attention_15/Reshape_3Reshape,multi_head_self_attention_15/transpose_3:y:05multi_head_self_attention_15/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2(
&multi_head_self_attention_15/Reshape_3?
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_138_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_138/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_138/Tensordot/axes?
5multi_head_self_attention_15/dense_138/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_138/Tensordot/free?
6multi_head_self_attention_15/dense_138/Tensordot/ShapeShape/multi_head_self_attention_15/Reshape_3:output:0*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_138/Tensordot/Shape?
>multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_138/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_138/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_138/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_138/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_138/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_138/Tensordot/Const?
5multi_head_self_attention_15/dense_138/Tensordot/ProdProdBmulti_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_138/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_138/Tensordot/Prod?
8multi_head_self_attention_15/dense_138/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_138/Tensordot/Const_1?
7multi_head_self_attention_15/dense_138/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_138/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_138/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_138/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_138/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_138/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_138/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_138/Tensordot/free:output:0>multi_head_self_attention_15/dense_138/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_138/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_138/Tensordot/concat?
6multi_head_self_attention_15/dense_138/Tensordot/stackPack>multi_head_self_attention_15/dense_138/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_138/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_138/Tensordot/stack?
:multi_head_self_attention_15/dense_138/Tensordot/transpose	Transpose/multi_head_self_attention_15/Reshape_3:output:0@multi_head_self_attention_15/dense_138/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2<
:multi_head_self_attention_15/dense_138/Tensordot/transpose?
8multi_head_self_attention_15/dense_138/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_138/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_138/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_138/Tensordot/Reshape?
7multi_head_self_attention_15/dense_138/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_138/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_138/Tensordot/MatMul?
8multi_head_self_attention_15/dense_138/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_138/Tensordot/Const_2?
>multi_head_self_attention_15/dense_138/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_138/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_138/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_138/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_138/Tensordot/concat_1?
0multi_head_self_attention_15/dense_138/TensordotReshapeAmulti_head_self_attention_15/dense_138/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_138/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 22
0multi_head_self_attention_15/dense_138/Tensordot?
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_138/BiasAddBiasAdd9multi_head_self_attention_15/dense_138/Tensordot:output:0Emulti_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_15/dense_138/BiasAddy
dropout_30/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_30/dropout/Const?
dropout_30/dropout/MulMul7multi_head_self_attention_15/dense_138/BiasAdd:output:0!dropout_30/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_30/dropout/Mul?
dropout_30/dropout/ShapeShape7multi_head_self_attention_15/dense_138/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_30/dropout/Shape?
/dropout_30/dropout/random_uniform/RandomUniformRandomUniform!dropout_30/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype021
/dropout_30/dropout/random_uniform/RandomUniform?
!dropout_30/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_30/dropout/GreaterEqual/y?
dropout_30/dropout/GreaterEqualGreaterEqual8dropout_30/dropout/random_uniform/RandomUniform:output:0*dropout_30/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2!
dropout_30/dropout/GreaterEqual?
dropout_30/dropout/CastCast#dropout_30/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout_30/dropout/Cast?
dropout_30/dropout/Mul_1Muldropout_30/dropout/Mul:z:0dropout_30/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_30/dropout/Mul_1o
addAddV2inputsdropout_30/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add?
5layer_normalization_30/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_30/moments/mean/reduction_indices?
#layer_normalization_30/moments/meanMeanadd:z:0>layer_normalization_30/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_30/moments/mean?
+layer_normalization_30/moments/StopGradientStopGradient,layer_normalization_30/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_30/moments/StopGradient?
0layer_normalization_30/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_30/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_30/moments/SquaredDifference?
9layer_normalization_30/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_30/moments/variance/reduction_indices?
'layer_normalization_30/moments/varianceMean4layer_normalization_30/moments/SquaredDifference:z:0Blayer_normalization_30/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_30/moments/variance?
&layer_normalization_30/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_30/batchnorm/add/y?
$layer_normalization_30/batchnorm/addAddV20layer_normalization_30/moments/variance:output:0/layer_normalization_30/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_30/batchnorm/add?
&layer_normalization_30/batchnorm/RsqrtRsqrt(layer_normalization_30/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_30/batchnorm/Rsqrt?
3layer_normalization_30/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_30_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_30/batchnorm/mul/ReadVariableOp?
$layer_normalization_30/batchnorm/mulMul*layer_normalization_30/batchnorm/Rsqrt:y:0;layer_normalization_30/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_30/batchnorm/mul?
&layer_normalization_30/batchnorm/mul_1Muladd:z:0(layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/mul_1?
&layer_normalization_30/batchnorm/mul_2Mul,layer_normalization_30/moments/mean:output:0(layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/mul_2?
/layer_normalization_30/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_30_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_30/batchnorm/ReadVariableOp?
$layer_normalization_30/batchnorm/subSub7layer_normalization_30/batchnorm/ReadVariableOp:value:0*layer_normalization_30/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_30/batchnorm/sub?
&layer_normalization_30/batchnorm/add_1AddV2*layer_normalization_30/batchnorm/mul_1:z:0(layer_normalization_30/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/add_1?
0sequential_15/dense_139/Tensordot/ReadVariableOpReadVariableOp9sequential_15_dense_139_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype022
0sequential_15/dense_139/Tensordot/ReadVariableOp?
&sequential_15/dense_139/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_15/dense_139/Tensordot/axes?
&sequential_15/dense_139/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&sequential_15/dense_139/Tensordot/free?
'sequential_15/dense_139/Tensordot/ShapeShape*layer_normalization_30/batchnorm/add_1:z:0*
T0*
_output_shapes
:2)
'sequential_15/dense_139/Tensordot/Shape?
/sequential_15/dense_139/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_139/Tensordot/GatherV2/axis?
*sequential_15/dense_139/Tensordot/GatherV2GatherV20sequential_15/dense_139/Tensordot/Shape:output:0/sequential_15/dense_139/Tensordot/free:output:08sequential_15/dense_139/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_15/dense_139/Tensordot/GatherV2?
1sequential_15/dense_139/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_15/dense_139/Tensordot/GatherV2_1/axis?
,sequential_15/dense_139/Tensordot/GatherV2_1GatherV20sequential_15/dense_139/Tensordot/Shape:output:0/sequential_15/dense_139/Tensordot/axes:output:0:sequential_15/dense_139/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,sequential_15/dense_139/Tensordot/GatherV2_1?
'sequential_15/dense_139/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_15/dense_139/Tensordot/Const?
&sequential_15/dense_139/Tensordot/ProdProd3sequential_15/dense_139/Tensordot/GatherV2:output:00sequential_15/dense_139/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&sequential_15/dense_139/Tensordot/Prod?
)sequential_15/dense_139/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_139/Tensordot/Const_1?
(sequential_15/dense_139/Tensordot/Prod_1Prod5sequential_15/dense_139/Tensordot/GatherV2_1:output:02sequential_15/dense_139/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(sequential_15/dense_139/Tensordot/Prod_1?
-sequential_15/dense_139/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_15/dense_139/Tensordot/concat/axis?
(sequential_15/dense_139/Tensordot/concatConcatV2/sequential_15/dense_139/Tensordot/free:output:0/sequential_15/dense_139/Tensordot/axes:output:06sequential_15/dense_139/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_15/dense_139/Tensordot/concat?
'sequential_15/dense_139/Tensordot/stackPack/sequential_15/dense_139/Tensordot/Prod:output:01sequential_15/dense_139/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_15/dense_139/Tensordot/stack?
+sequential_15/dense_139/Tensordot/transpose	Transpose*layer_normalization_30/batchnorm/add_1:z:01sequential_15/dense_139/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2-
+sequential_15/dense_139/Tensordot/transpose?
)sequential_15/dense_139/Tensordot/ReshapeReshape/sequential_15/dense_139/Tensordot/transpose:y:00sequential_15/dense_139/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)sequential_15/dense_139/Tensordot/Reshape?
(sequential_15/dense_139/Tensordot/MatMulMatMul2sequential_15/dense_139/Tensordot/Reshape:output:08sequential_15/dense_139/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(sequential_15/dense_139/Tensordot/MatMul?
)sequential_15/dense_139/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_139/Tensordot/Const_2?
/sequential_15/dense_139/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_139/Tensordot/concat_1/axis?
*sequential_15/dense_139/Tensordot/concat_1ConcatV23sequential_15/dense_139/Tensordot/GatherV2:output:02sequential_15/dense_139/Tensordot/Const_2:output:08sequential_15/dense_139/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*sequential_15/dense_139/Tensordot/concat_1?
!sequential_15/dense_139/TensordotReshape2sequential_15/dense_139/Tensordot/MatMul:product:03sequential_15/dense_139/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2#
!sequential_15/dense_139/Tensordot?
.sequential_15/dense_139/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_dense_139_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/dense_139/BiasAdd/ReadVariableOp?
sequential_15/dense_139/BiasAddBiasAdd*sequential_15/dense_139/Tensordot:output:06sequential_15/dense_139/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2!
sequential_15/dense_139/BiasAdd?
sequential_15/dense_139/ReluRelu(sequential_15/dense_139/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_15/dense_139/Relu?
0sequential_15/dense_140/Tensordot/ReadVariableOpReadVariableOp9sequential_15_dense_140_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype022
0sequential_15/dense_140/Tensordot/ReadVariableOp?
&sequential_15/dense_140/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_15/dense_140/Tensordot/axes?
&sequential_15/dense_140/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&sequential_15/dense_140/Tensordot/free?
'sequential_15/dense_140/Tensordot/ShapeShape*sequential_15/dense_139/Relu:activations:0*
T0*
_output_shapes
:2)
'sequential_15/dense_140/Tensordot/Shape?
/sequential_15/dense_140/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_140/Tensordot/GatherV2/axis?
*sequential_15/dense_140/Tensordot/GatherV2GatherV20sequential_15/dense_140/Tensordot/Shape:output:0/sequential_15/dense_140/Tensordot/free:output:08sequential_15/dense_140/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_15/dense_140/Tensordot/GatherV2?
1sequential_15/dense_140/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_15/dense_140/Tensordot/GatherV2_1/axis?
,sequential_15/dense_140/Tensordot/GatherV2_1GatherV20sequential_15/dense_140/Tensordot/Shape:output:0/sequential_15/dense_140/Tensordot/axes:output:0:sequential_15/dense_140/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,sequential_15/dense_140/Tensordot/GatherV2_1?
'sequential_15/dense_140/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_15/dense_140/Tensordot/Const?
&sequential_15/dense_140/Tensordot/ProdProd3sequential_15/dense_140/Tensordot/GatherV2:output:00sequential_15/dense_140/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&sequential_15/dense_140/Tensordot/Prod?
)sequential_15/dense_140/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_140/Tensordot/Const_1?
(sequential_15/dense_140/Tensordot/Prod_1Prod5sequential_15/dense_140/Tensordot/GatherV2_1:output:02sequential_15/dense_140/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(sequential_15/dense_140/Tensordot/Prod_1?
-sequential_15/dense_140/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_15/dense_140/Tensordot/concat/axis?
(sequential_15/dense_140/Tensordot/concatConcatV2/sequential_15/dense_140/Tensordot/free:output:0/sequential_15/dense_140/Tensordot/axes:output:06sequential_15/dense_140/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_15/dense_140/Tensordot/concat?
'sequential_15/dense_140/Tensordot/stackPack/sequential_15/dense_140/Tensordot/Prod:output:01sequential_15/dense_140/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_15/dense_140/Tensordot/stack?
+sequential_15/dense_140/Tensordot/transpose	Transpose*sequential_15/dense_139/Relu:activations:01sequential_15/dense_140/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2-
+sequential_15/dense_140/Tensordot/transpose?
)sequential_15/dense_140/Tensordot/ReshapeReshape/sequential_15/dense_140/Tensordot/transpose:y:00sequential_15/dense_140/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)sequential_15/dense_140/Tensordot/Reshape?
(sequential_15/dense_140/Tensordot/MatMulMatMul2sequential_15/dense_140/Tensordot/Reshape:output:08sequential_15/dense_140/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(sequential_15/dense_140/Tensordot/MatMul?
)sequential_15/dense_140/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_140/Tensordot/Const_2?
/sequential_15/dense_140/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_140/Tensordot/concat_1/axis?
*sequential_15/dense_140/Tensordot/concat_1ConcatV23sequential_15/dense_140/Tensordot/GatherV2:output:02sequential_15/dense_140/Tensordot/Const_2:output:08sequential_15/dense_140/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*sequential_15/dense_140/Tensordot/concat_1?
!sequential_15/dense_140/TensordotReshape2sequential_15/dense_140/Tensordot/MatMul:product:03sequential_15/dense_140/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2#
!sequential_15/dense_140/Tensordot?
.sequential_15/dense_140/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/dense_140/BiasAdd/ReadVariableOp?
sequential_15/dense_140/BiasAddBiasAdd*sequential_15/dense_140/Tensordot:output:06sequential_15/dense_140/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2!
sequential_15/dense_140/BiasAddy
dropout_31/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_31/dropout/Const?
dropout_31/dropout/MulMul(sequential_15/dense_140/BiasAdd:output:0!dropout_31/dropout/Const:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_31/dropout/Mul?
dropout_31/dropout/ShapeShape(sequential_15/dense_140/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_31/dropout/Shape?
/dropout_31/dropout/random_uniform/RandomUniformRandomUniform!dropout_31/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????( *
dtype021
/dropout_31/dropout/random_uniform/RandomUniform?
!dropout_31/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_31/dropout/GreaterEqual/y?
dropout_31/dropout/GreaterEqualGreaterEqual8dropout_31/dropout/random_uniform/RandomUniform:output:0*dropout_31/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????( 2!
dropout_31/dropout/GreaterEqual?
dropout_31/dropout/CastCast#dropout_31/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????( 2
dropout_31/dropout/Cast?
dropout_31/dropout/Mul_1Muldropout_31/dropout/Mul:z:0dropout_31/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????( 2
dropout_31/dropout/Mul_1?
add_1AddV2*layer_normalization_30/batchnorm/add_1:z:0dropout_31/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add_1?
5layer_normalization_31/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_31/moments/mean/reduction_indices?
#layer_normalization_31/moments/meanMean	add_1:z:0>layer_normalization_31/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_31/moments/mean?
+layer_normalization_31/moments/StopGradientStopGradient,layer_normalization_31/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_31/moments/StopGradient?
0layer_normalization_31/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_31/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_31/moments/SquaredDifference?
9layer_normalization_31/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_31/moments/variance/reduction_indices?
'layer_normalization_31/moments/varianceMean4layer_normalization_31/moments/SquaredDifference:z:0Blayer_normalization_31/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_31/moments/variance?
&layer_normalization_31/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_31/batchnorm/add/y?
$layer_normalization_31/batchnorm/addAddV20layer_normalization_31/moments/variance:output:0/layer_normalization_31/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_31/batchnorm/add?
&layer_normalization_31/batchnorm/RsqrtRsqrt(layer_normalization_31/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_31/batchnorm/Rsqrt?
3layer_normalization_31/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_31_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_31/batchnorm/mul/ReadVariableOp?
$layer_normalization_31/batchnorm/mulMul*layer_normalization_31/batchnorm/Rsqrt:y:0;layer_normalization_31/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_31/batchnorm/mul?
&layer_normalization_31/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/mul_1?
&layer_normalization_31/batchnorm/mul_2Mul,layer_normalization_31/moments/mean:output:0(layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/mul_2?
/layer_normalization_31/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_31_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_31/batchnorm/ReadVariableOp?
$layer_normalization_31/batchnorm/subSub7layer_normalization_31/batchnorm/ReadVariableOp:value:0*layer_normalization_31/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_31/batchnorm/sub?
&layer_normalization_31/batchnorm/add_1AddV2*layer_normalization_31/batchnorm/mul_1:z:0(layer_normalization_31/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/add_1?
IdentityIdentity*layer_normalization_31/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp0^layer_normalization_30/batchnorm/ReadVariableOp4^layer_normalization_30/batchnorm/mul/ReadVariableOp0^layer_normalization_31/batchnorm/ReadVariableOp4^layer_normalization_31/batchnorm/mul/ReadVariableOp>^multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp/^sequential_15/dense_139/BiasAdd/ReadVariableOp1^sequential_15/dense_139/Tensordot/ReadVariableOp/^sequential_15/dense_140/BiasAdd/ReadVariableOp1^sequential_15/dense_140/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2b
/layer_normalization_30/batchnorm/ReadVariableOp/layer_normalization_30/batchnorm/ReadVariableOp2j
3layer_normalization_30/batchnorm/mul/ReadVariableOp3layer_normalization_30/batchnorm/mul/ReadVariableOp2b
/layer_normalization_31/batchnorm/ReadVariableOp/layer_normalization_31/batchnorm/ReadVariableOp2j
3layer_normalization_31/batchnorm/mul/ReadVariableOp3layer_normalization_31/batchnorm/mul/ReadVariableOp2~
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp2`
.sequential_15/dense_139/BiasAdd/ReadVariableOp.sequential_15/dense_139/BiasAdd/ReadVariableOp2d
0sequential_15/dense_139/Tensordot/ReadVariableOp0sequential_15/dense_139/Tensordot/ReadVariableOp2`
.sequential_15/dense_140/BiasAdd/ReadVariableOp.sequential_15/dense_140/BiasAdd/ReadVariableOp2d
0sequential_15/dense_140/Tensordot/ReadVariableOp0sequential_15/dense_140/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
? 
?
F__inference_dense_140_layer_call_and_return_conditional_losses_6975037

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????( : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
??
?!
E__inference_model_15_layer_call_and_return_conditional_losses_6977180
inputs_0
inputs_1
inputs_2W
Etoken_and_position_embedding_15_embedding_31_embedding_lookup_6976873:( W
Etoken_and_position_embedding_15_embedding_30_embedding_lookup_6976879: o
]transformer_block_15_multi_head_self_attention_15_dense_135_tensordot_readvariableop_resource:  i
[transformer_block_15_multi_head_self_attention_15_dense_135_biasadd_readvariableop_resource: o
]transformer_block_15_multi_head_self_attention_15_dense_136_tensordot_readvariableop_resource:  i
[transformer_block_15_multi_head_self_attention_15_dense_136_biasadd_readvariableop_resource: o
]transformer_block_15_multi_head_self_attention_15_dense_137_tensordot_readvariableop_resource:  i
[transformer_block_15_multi_head_self_attention_15_dense_137_biasadd_readvariableop_resource: o
]transformer_block_15_multi_head_self_attention_15_dense_138_tensordot_readvariableop_resource:  i
[transformer_block_15_multi_head_self_attention_15_dense_138_biasadd_readvariableop_resource: _
Qtransformer_block_15_layer_normalization_30_batchnorm_mul_readvariableop_resource: [
Mtransformer_block_15_layer_normalization_30_batchnorm_readvariableop_resource: `
Ntransformer_block_15_sequential_15_dense_139_tensordot_readvariableop_resource:  Z
Ltransformer_block_15_sequential_15_dense_139_biasadd_readvariableop_resource: `
Ntransformer_block_15_sequential_15_dense_140_tensordot_readvariableop_resource:  Z
Ltransformer_block_15_sequential_15_dense_140_biasadd_readvariableop_resource: _
Qtransformer_block_15_layer_normalization_31_batchnorm_mul_readvariableop_resource: [
Mtransformer_block_15_layer_normalization_31_batchnorm_readvariableop_resource: ;
)aux_output_matmul_readvariableop_resource: 8
*aux_output_biasadd_readvariableop_resource::
(dense_141_matmul_readvariableop_resource:@7
)dense_141_biasadd_readvariableop_resource:@:
(dense_142_matmul_readvariableop_resource:@@7
)dense_142_biasadd_readvariableop_resource:@:
(dense_143_matmul_readvariableop_resource:@@7
)dense_143_biasadd_readvariableop_resource:@<
*main_output_matmul_readvariableop_resource:@9
+main_output_biasadd_readvariableop_resource:
identity

identity_1??!aux_output/BiasAdd/ReadVariableOp? aux_output/MatMul/ReadVariableOp? dense_141/BiasAdd/ReadVariableOp?dense_141/MatMul/ReadVariableOp? dense_142/BiasAdd/ReadVariableOp?dense_142/MatMul/ReadVariableOp? dense_143/BiasAdd/ReadVariableOp?dense_143/MatMul/ReadVariableOp?"main_output/BiasAdd/ReadVariableOp?!main_output/MatMul/ReadVariableOp?=token_and_position_embedding_15/embedding_30/embedding_lookup?=token_and_position_embedding_15/embedding_31/embedding_lookup?Dtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp?Htransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp?Dtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp?Htransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp?Rtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp?Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?Rtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp?Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?Rtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp?Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?Rtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp?Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?Ctransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp?Etransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp?Ctransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp?Etransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp?
%token_and_position_embedding_15/ShapeShapeinputs_0*
T0*
_output_shapes
:2'
%token_and_position_embedding_15/Shape?
3token_and_position_embedding_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????25
3token_and_position_embedding_15/strided_slice/stack?
5token_and_position_embedding_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5token_and_position_embedding_15/strided_slice/stack_1?
5token_and_position_embedding_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5token_and_position_embedding_15/strided_slice/stack_2?
-token_and_position_embedding_15/strided_sliceStridedSlice.token_and_position_embedding_15/Shape:output:0<token_and_position_embedding_15/strided_slice/stack:output:0>token_and_position_embedding_15/strided_slice/stack_1:output:0>token_and_position_embedding_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-token_and_position_embedding_15/strided_slice?
+token_and_position_embedding_15/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2-
+token_and_position_embedding_15/range/start?
+token_and_position_embedding_15/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2-
+token_and_position_embedding_15/range/delta?
%token_and_position_embedding_15/rangeRange4token_and_position_embedding_15/range/start:output:06token_and_position_embedding_15/strided_slice:output:04token_and_position_embedding_15/range/delta:output:0*#
_output_shapes
:?????????2'
%token_and_position_embedding_15/range?
=token_and_position_embedding_15/embedding_31/embedding_lookupResourceGatherEtoken_and_position_embedding_15_embedding_31_embedding_lookup_6976873.token_and_position_embedding_15/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*X
_classN
LJloc:@token_and_position_embedding_15/embedding_31/embedding_lookup/6976873*'
_output_shapes
:????????? *
dtype02?
=token_and_position_embedding_15/embedding_31/embedding_lookup?
Ftoken_and_position_embedding_15/embedding_31/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_15/embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@token_and_position_embedding_15/embedding_31/embedding_lookup/6976873*'
_output_shapes
:????????? 2H
Ftoken_and_position_embedding_15/embedding_31/embedding_lookup/Identity?
Htoken_and_position_embedding_15/embedding_31/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_15/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2J
Htoken_and_position_embedding_15/embedding_31/embedding_lookup/Identity_1?
1token_and_position_embedding_15/embedding_30/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????(23
1token_and_position_embedding_15/embedding_30/Cast?
=token_and_position_embedding_15/embedding_30/embedding_lookupResourceGatherEtoken_and_position_embedding_15_embedding_30_embedding_lookup_69768795token_and_position_embedding_15/embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*X
_classN
LJloc:@token_and_position_embedding_15/embedding_30/embedding_lookup/6976879*+
_output_shapes
:?????????( *
dtype02?
=token_and_position_embedding_15/embedding_30/embedding_lookup?
Ftoken_and_position_embedding_15/embedding_30/embedding_lookup/IdentityIdentityFtoken_and_position_embedding_15/embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@token_and_position_embedding_15/embedding_30/embedding_lookup/6976879*+
_output_shapes
:?????????( 2H
Ftoken_and_position_embedding_15/embedding_30/embedding_lookup/Identity?
Htoken_and_position_embedding_15/embedding_30/embedding_lookup/Identity_1IdentityOtoken_and_position_embedding_15/embedding_30/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2J
Htoken_and_position_embedding_15/embedding_30/embedding_lookup/Identity_1?
#token_and_position_embedding_15/addAddV2Qtoken_and_position_embedding_15/embedding_30/embedding_lookup/Identity_1:output:0Qtoken_and_position_embedding_15/embedding_31/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2%
#token_and_position_embedding_15/add?
7transformer_block_15/multi_head_self_attention_15/ShapeShape'token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:29
7transformer_block_15/multi_head_self_attention_15/Shape?
Etransformer_block_15/multi_head_self_attention_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block_15/multi_head_self_attention_15/strided_slice/stack?
Gtransformer_block_15/multi_head_self_attention_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_15/multi_head_self_attention_15/strided_slice/stack_1?
Gtransformer_block_15/multi_head_self_attention_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_15/multi_head_self_attention_15/strided_slice/stack_2?
?transformer_block_15/multi_head_self_attention_15/strided_sliceStridedSlice@transformer_block_15/multi_head_self_attention_15/Shape:output:0Ntransformer_block_15/multi_head_self_attention_15/strided_slice/stack:output:0Ptransformer_block_15/multi_head_self_attention_15/strided_slice/stack_1:output:0Ptransformer_block_15/multi_head_self_attention_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?transformer_block_15/multi_head_self_attention_15/strided_slice?
Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpReadVariableOp]transformer_block_15_multi_head_self_attention_15_dense_135_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02V
Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axes?
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/free?
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ShapeShape'token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Shape?
Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/free:output:0\transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2?
Utransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Utransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis?
Ptransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axes:output:0^transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Ptransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1?
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const?
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ProdProdWtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod?
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_1?
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod_1ProdYtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod_1?
Qtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat/axis?
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concatConcatV2Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/free:output:0Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axes:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat?
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/stackPackStransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod:output:0Utransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/stack?
Otransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/transpose	Transpose'token_and_position_embedding_15/add:z:0Utransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2Q
Otransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/transpose?
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReshapeReshapeStransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/transpose:y:0Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Reshape?
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/MatMulMatMulVtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Reshape:output:0\transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/MatMul?
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_2?
Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1ConcatV2Wtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_2:output:0\transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1?
Etransformer_block_15/multi_head_self_attention_15/dense_135/TensordotReshapeVtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/MatMul:product:0Wtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2G
Etransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot?
Rtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpReadVariableOp[transformer_block_15_multi_head_self_attention_15_dense_135_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp?
Ctransformer_block_15/multi_head_self_attention_15/dense_135/BiasAddBiasAddNtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2E
Ctransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd?
Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpReadVariableOp]transformer_block_15_multi_head_self_attention_15_dense_136_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02V
Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axes?
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/free?
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ShapeShape'token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Shape?
Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/free:output:0\transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2?
Utransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Utransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis?
Ptransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axes:output:0^transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Ptransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1?
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const?
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ProdProdWtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod?
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_1?
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod_1ProdYtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod_1?
Qtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat/axis?
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concatConcatV2Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/free:output:0Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axes:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat?
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/stackPackStransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod:output:0Utransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/stack?
Otransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/transpose	Transpose'token_and_position_embedding_15/add:z:0Utransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2Q
Otransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/transpose?
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReshapeReshapeStransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/transpose:y:0Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Reshape?
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/MatMulMatMulVtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Reshape:output:0\transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/MatMul?
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_2?
Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1ConcatV2Wtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_2:output:0\transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1?
Etransformer_block_15/multi_head_self_attention_15/dense_136/TensordotReshapeVtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/MatMul:product:0Wtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2G
Etransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot?
Rtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpReadVariableOp[transformer_block_15_multi_head_self_attention_15_dense_136_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp?
Ctransformer_block_15/multi_head_self_attention_15/dense_136/BiasAddBiasAddNtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2E
Ctransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd?
Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpReadVariableOp]transformer_block_15_multi_head_self_attention_15_dense_137_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02V
Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axes?
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/free?
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ShapeShape'token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Shape?
Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/free:output:0\transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2?
Utransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Utransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis?
Ptransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axes:output:0^transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Ptransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1?
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const?
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ProdProdWtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod?
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_1?
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod_1ProdYtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod_1?
Qtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat/axis?
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concatConcatV2Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/free:output:0Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axes:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat?
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/stackPackStransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod:output:0Utransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/stack?
Otransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/transpose	Transpose'token_and_position_embedding_15/add:z:0Utransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2Q
Otransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/transpose?
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReshapeReshapeStransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/transpose:y:0Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Reshape?
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/MatMulMatMulVtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Reshape:output:0\transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/MatMul?
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_2?
Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1ConcatV2Wtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_2:output:0\transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1?
Etransformer_block_15/multi_head_self_attention_15/dense_137/TensordotReshapeVtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/MatMul:product:0Wtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2G
Etransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot?
Rtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpReadVariableOp[transformer_block_15_multi_head_self_attention_15_dense_137_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp?
Ctransformer_block_15/multi_head_self_attention_15/dense_137/BiasAddBiasAddNtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2E
Ctransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd?
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/1?
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/2?
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_15/multi_head_self_attention_15/Reshape/shape/3?
?transformer_block_15/multi_head_self_attention_15/Reshape/shapePackHtransformer_block_15/multi_head_self_attention_15/strided_slice:output:0Jtransformer_block_15/multi_head_self_attention_15/Reshape/shape/1:output:0Jtransformer_block_15/multi_head_self_attention_15/Reshape/shape/2:output:0Jtransformer_block_15/multi_head_self_attention_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_15/multi_head_self_attention_15/Reshape/shape?
9transformer_block_15/multi_head_self_attention_15/ReshapeReshapeLtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd:output:0Htransformer_block_15/multi_head_self_attention_15/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_15/multi_head_self_attention_15/Reshape?
@transformer_block_15/multi_head_self_attention_15/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_15/multi_head_self_attention_15/transpose/perm?
;transformer_block_15/multi_head_self_attention_15/transpose	TransposeBtransformer_block_15/multi_head_self_attention_15/Reshape:output:0Itransformer_block_15/multi_head_self_attention_15/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_15/multi_head_self_attention_15/transpose?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/1?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/2?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/3?
Atransformer_block_15/multi_head_self_attention_15/Reshape_1/shapePackHtransformer_block_15/multi_head_self_attention_15/strided_slice:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/1:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/2:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2C
Atransformer_block_15/multi_head_self_attention_15/Reshape_1/shape?
;transformer_block_15/multi_head_self_attention_15/Reshape_1ReshapeLtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd:output:0Jtransformer_block_15/multi_head_self_attention_15/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_15/multi_head_self_attention_15/Reshape_1?
Btransformer_block_15/multi_head_self_attention_15/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2D
Btransformer_block_15/multi_head_self_attention_15/transpose_1/perm?
=transformer_block_15/multi_head_self_attention_15/transpose_1	TransposeDtransformer_block_15/multi_head_self_attention_15/Reshape_1:output:0Ktransformer_block_15/multi_head_self_attention_15/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2?
=transformer_block_15/multi_head_self_attention_15/transpose_1?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/1?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/2?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/3?
Atransformer_block_15/multi_head_self_attention_15/Reshape_2/shapePackHtransformer_block_15/multi_head_self_attention_15/strided_slice:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/1:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/2:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2C
Atransformer_block_15/multi_head_self_attention_15/Reshape_2/shape?
;transformer_block_15/multi_head_self_attention_15/Reshape_2ReshapeLtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd:output:0Jtransformer_block_15/multi_head_self_attention_15/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_15/multi_head_self_attention_15/Reshape_2?
Btransformer_block_15/multi_head_self_attention_15/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2D
Btransformer_block_15/multi_head_self_attention_15/transpose_2/perm?
=transformer_block_15/multi_head_self_attention_15/transpose_2	TransposeDtransformer_block_15/multi_head_self_attention_15/Reshape_2:output:0Ktransformer_block_15/multi_head_self_attention_15/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2?
=transformer_block_15/multi_head_self_attention_15/transpose_2?
8transformer_block_15/multi_head_self_attention_15/MatMulBatchMatMulV2?transformer_block_15/multi_head_self_attention_15/transpose:y:0Atransformer_block_15/multi_head_self_attention_15/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2:
8transformer_block_15/multi_head_self_attention_15/MatMul?
9transformer_block_15/multi_head_self_attention_15/Shape_1ShapeAtransformer_block_15/multi_head_self_attention_15/transpose_1:y:0*
T0*
_output_shapes
:2;
9transformer_block_15/multi_head_self_attention_15/Shape_1?
Gtransformer_block_15/multi_head_self_attention_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2I
Gtransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack?
Itransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2K
Itransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_1?
Itransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_2?
Atransformer_block_15/multi_head_self_attention_15/strided_slice_1StridedSliceBtransformer_block_15/multi_head_self_attention_15/Shape_1:output:0Ptransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack:output:0Rtransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_1:output:0Rtransformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Atransformer_block_15/multi_head_self_attention_15/strided_slice_1?
6transformer_block_15/multi_head_self_attention_15/CastCastJtransformer_block_15/multi_head_self_attention_15/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 28
6transformer_block_15/multi_head_self_attention_15/Cast?
6transformer_block_15/multi_head_self_attention_15/SqrtSqrt:transformer_block_15/multi_head_self_attention_15/Cast:y:0*
T0*
_output_shapes
: 28
6transformer_block_15/multi_head_self_attention_15/Sqrt?
9transformer_block_15/multi_head_self_attention_15/truedivRealDivAtransformer_block_15/multi_head_self_attention_15/MatMul:output:0:transformer_block_15/multi_head_self_attention_15/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2;
9transformer_block_15/multi_head_self_attention_15/truediv?
9transformer_block_15/multi_head_self_attention_15/SoftmaxSoftmax=transformer_block_15/multi_head_self_attention_15/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2;
9transformer_block_15/multi_head_self_attention_15/Softmax?
:transformer_block_15/multi_head_self_attention_15/MatMul_1BatchMatMulV2Ctransformer_block_15/multi_head_self_attention_15/Softmax:softmax:0Atransformer_block_15/multi_head_self_attention_15/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2<
:transformer_block_15/multi_head_self_attention_15/MatMul_1?
Btransformer_block_15/multi_head_self_attention_15/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2D
Btransformer_block_15/multi_head_self_attention_15/transpose_3/perm?
=transformer_block_15/multi_head_self_attention_15/transpose_3	TransposeCtransformer_block_15/multi_head_self_attention_15/MatMul_1:output:0Ktransformer_block_15/multi_head_self_attention_15/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2?
=transformer_block_15/multi_head_self_attention_15/transpose_3?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/1?
Ctransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/2?
Atransformer_block_15/multi_head_self_attention_15/Reshape_3/shapePackHtransformer_block_15/multi_head_self_attention_15/strided_slice:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/1:output:0Ltransformer_block_15/multi_head_self_attention_15/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2C
Atransformer_block_15/multi_head_self_attention_15/Reshape_3/shape?
;transformer_block_15/multi_head_self_attention_15/Reshape_3ReshapeAtransformer_block_15/multi_head_self_attention_15/transpose_3:y:0Jtransformer_block_15/multi_head_self_attention_15/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2=
;transformer_block_15/multi_head_self_attention_15/Reshape_3?
Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpReadVariableOp]transformer_block_15_multi_head_self_attention_15_dense_138_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02V
Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axes?
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/free?
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ShapeShapeDtransformer_block_15/multi_head_self_attention_15/Reshape_3:output:0*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Shape?
Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/free:output:0\transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2?
Utransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Utransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis?
Ptransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1GatherV2Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axes:output:0^transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Ptransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1?
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const?
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ProdProdWtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod?
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_1?
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod_1ProdYtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod_1?
Qtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat/axis?
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concatConcatV2Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/free:output:0Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axes:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat?
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/stackPackStransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod:output:0Utransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/stack?
Otransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/transpose	TransposeDtransformer_block_15/multi_head_self_attention_15/Reshape_3:output:0Utransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2Q
Otransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/transpose?
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReshapeReshapeStransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/transpose:y:0Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Reshape?
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/MatMulMatMulVtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Reshape:output:0\transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2N
Ltransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/MatMul?
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_2?
Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Stransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1/axis?
Ntransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1ConcatV2Wtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0Vtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_2:output:0\transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Ntransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1?
Etransformer_block_15/multi_head_self_attention_15/dense_138/TensordotReshapeVtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/MatMul:product:0Wtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 2G
Etransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot?
Rtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpReadVariableOp[transformer_block_15_multi_head_self_attention_15_dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02T
Rtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp?
Ctransformer_block_15/multi_head_self_attention_15/dense_138/BiasAddBiasAddNtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot:output:0Ztransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2E
Ctransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd?
-transformer_block_15/dropout_30/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2/
-transformer_block_15/dropout_30/dropout/Const?
+transformer_block_15/dropout_30/dropout/MulMulLtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd:output:06transformer_block_15/dropout_30/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2-
+transformer_block_15/dropout_30/dropout/Mul?
-transformer_block_15/dropout_30/dropout/ShapeShapeLtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd:output:0*
T0*
_output_shapes
:2/
-transformer_block_15/dropout_30/dropout/Shape?
Dtransformer_block_15/dropout_30/dropout/random_uniform/RandomUniformRandomUniform6transformer_block_15/dropout_30/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype02F
Dtransformer_block_15/dropout_30/dropout/random_uniform/RandomUniform?
6transformer_block_15/dropout_30/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=28
6transformer_block_15/dropout_30/dropout/GreaterEqual/y?
4transformer_block_15/dropout_30/dropout/GreaterEqualGreaterEqualMtransformer_block_15/dropout_30/dropout/random_uniform/RandomUniform:output:0?transformer_block_15/dropout_30/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 26
4transformer_block_15/dropout_30/dropout/GreaterEqual?
,transformer_block_15/dropout_30/dropout/CastCast8transformer_block_15/dropout_30/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2.
,transformer_block_15/dropout_30/dropout/Cast?
-transformer_block_15/dropout_30/dropout/Mul_1Mul/transformer_block_15/dropout_30/dropout/Mul:z:00transformer_block_15/dropout_30/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2/
-transformer_block_15/dropout_30/dropout/Mul_1?
transformer_block_15/addAddV2'token_and_position_embedding_15/add:z:01transformer_block_15/dropout_30/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_15/add?
Jtransformer_block_15/layer_normalization_30/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/layer_normalization_30/moments/mean/reduction_indices?
8transformer_block_15/layer_normalization_30/moments/meanMeantransformer_block_15/add:z:0Stransformer_block_15/layer_normalization_30/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2:
8transformer_block_15/layer_normalization_30/moments/mean?
@transformer_block_15/layer_normalization_30/moments/StopGradientStopGradientAtransformer_block_15/layer_normalization_30/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2B
@transformer_block_15/layer_normalization_30/moments/StopGradient?
Etransformer_block_15/layer_normalization_30/moments/SquaredDifferenceSquaredDifferencetransformer_block_15/add:z:0Itransformer_block_15/layer_normalization_30/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2G
Etransformer_block_15/layer_normalization_30/moments/SquaredDifference?
Ntransformer_block_15/layer_normalization_30/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Ntransformer_block_15/layer_normalization_30/moments/variance/reduction_indices?
<transformer_block_15/layer_normalization_30/moments/varianceMeanItransformer_block_15/layer_normalization_30/moments/SquaredDifference:z:0Wtransformer_block_15/layer_normalization_30/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2>
<transformer_block_15/layer_normalization_30/moments/variance?
;transformer_block_15/layer_normalization_30/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52=
;transformer_block_15/layer_normalization_30/batchnorm/add/y?
9transformer_block_15/layer_normalization_30/batchnorm/addAddV2Etransformer_block_15/layer_normalization_30/moments/variance:output:0Dtransformer_block_15/layer_normalization_30/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2;
9transformer_block_15/layer_normalization_30/batchnorm/add?
;transformer_block_15/layer_normalization_30/batchnorm/RsqrtRsqrt=transformer_block_15/layer_normalization_30/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2=
;transformer_block_15/layer_normalization_30/batchnorm/Rsqrt?
Htransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_block_15_layer_normalization_30_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02J
Htransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp?
9transformer_block_15/layer_normalization_30/batchnorm/mulMul?transformer_block_15/layer_normalization_30/batchnorm/Rsqrt:y:0Ptransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_15/layer_normalization_30/batchnorm/mul?
;transformer_block_15/layer_normalization_30/batchnorm/mul_1Multransformer_block_15/add:z:0=transformer_block_15/layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_30/batchnorm/mul_1?
;transformer_block_15/layer_normalization_30/batchnorm/mul_2MulAtransformer_block_15/layer_normalization_30/moments/mean:output:0=transformer_block_15/layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_30/batchnorm/mul_2?
Dtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOpReadVariableOpMtransformer_block_15_layer_normalization_30_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02F
Dtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp?
9transformer_block_15/layer_normalization_30/batchnorm/subSubLtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp:value:0?transformer_block_15/layer_normalization_30/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_15/layer_normalization_30/batchnorm/sub?
;transformer_block_15/layer_normalization_30/batchnorm/add_1AddV2?transformer_block_15/layer_normalization_30/batchnorm/mul_1:z:0=transformer_block_15/layer_normalization_30/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_30/batchnorm/add_1?
Etransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOpReadVariableOpNtransformer_block_15_sequential_15_dense_139_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02G
Etransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp?
;transformer_block_15/sequential_15/dense_139/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block_15/sequential_15/dense_139/Tensordot/axes?
;transformer_block_15/sequential_15/dense_139/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2=
;transformer_block_15/sequential_15/dense_139/Tensordot/free?
<transformer_block_15/sequential_15/dense_139/Tensordot/ShapeShape?transformer_block_15/layer_normalization_30/batchnorm/add_1:z:0*
T0*
_output_shapes
:2>
<transformer_block_15/sequential_15/dense_139/Tensordot/Shape?
Dtransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2/axis?
?transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2GatherV2Etransformer_block_15/sequential_15/dense_139/Tensordot/Shape:output:0Dtransformer_block_15/sequential_15/dense_139/Tensordot/free:output:0Mtransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2A
?transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2?
Ftransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Ftransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1/axis?
Atransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1GatherV2Etransformer_block_15/sequential_15/dense_139/Tensordot/Shape:output:0Dtransformer_block_15/sequential_15/dense_139/Tensordot/axes:output:0Otransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Atransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1?
<transformer_block_15/sequential_15/dense_139/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<transformer_block_15/sequential_15/dense_139/Tensordot/Const?
;transformer_block_15/sequential_15/dense_139/Tensordot/ProdProdHtransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2:output:0Etransformer_block_15/sequential_15/dense_139/Tensordot/Const:output:0*
T0*
_output_shapes
: 2=
;transformer_block_15/sequential_15/dense_139/Tensordot/Prod?
>transformer_block_15/sequential_15/dense_139/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>transformer_block_15/sequential_15/dense_139/Tensordot/Const_1?
=transformer_block_15/sequential_15/dense_139/Tensordot/Prod_1ProdJtransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1:output:0Gtransformer_block_15/sequential_15/dense_139/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2?
=transformer_block_15/sequential_15/dense_139/Tensordot/Prod_1?
Btransformer_block_15/sequential_15/dense_139/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_15/sequential_15/dense_139/Tensordot/concat/axis?
=transformer_block_15/sequential_15/dense_139/Tensordot/concatConcatV2Dtransformer_block_15/sequential_15/dense_139/Tensordot/free:output:0Dtransformer_block_15/sequential_15/dense_139/Tensordot/axes:output:0Ktransformer_block_15/sequential_15/dense_139/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_15/sequential_15/dense_139/Tensordot/concat?
<transformer_block_15/sequential_15/dense_139/Tensordot/stackPackDtransformer_block_15/sequential_15/dense_139/Tensordot/Prod:output:0Ftransformer_block_15/sequential_15/dense_139/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_15/sequential_15/dense_139/Tensordot/stack?
@transformer_block_15/sequential_15/dense_139/Tensordot/transpose	Transpose?transformer_block_15/layer_normalization_30/batchnorm/add_1:z:0Ftransformer_block_15/sequential_15/dense_139/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_15/sequential_15/dense_139/Tensordot/transpose?
>transformer_block_15/sequential_15/dense_139/Tensordot/ReshapeReshapeDtransformer_block_15/sequential_15/dense_139/Tensordot/transpose:y:0Etransformer_block_15/sequential_15/dense_139/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2@
>transformer_block_15/sequential_15/dense_139/Tensordot/Reshape?
=transformer_block_15/sequential_15/dense_139/Tensordot/MatMulMatMulGtransformer_block_15/sequential_15/dense_139/Tensordot/Reshape:output:0Mtransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2?
=transformer_block_15/sequential_15/dense_139/Tensordot/MatMul?
>transformer_block_15/sequential_15/dense_139/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2@
>transformer_block_15/sequential_15/dense_139/Tensordot/Const_2?
Dtransformer_block_15/sequential_15/dense_139/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_15/sequential_15/dense_139/Tensordot/concat_1/axis?
?transformer_block_15/sequential_15/dense_139/Tensordot/concat_1ConcatV2Htransformer_block_15/sequential_15/dense_139/Tensordot/GatherV2:output:0Gtransformer_block_15/sequential_15/dense_139/Tensordot/Const_2:output:0Mtransformer_block_15/sequential_15/dense_139/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_15/sequential_15/dense_139/Tensordot/concat_1?
6transformer_block_15/sequential_15/dense_139/TensordotReshapeGtransformer_block_15/sequential_15/dense_139/Tensordot/MatMul:product:0Htransformer_block_15/sequential_15/dense_139/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 28
6transformer_block_15/sequential_15/dense_139/Tensordot?
Ctransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOpReadVariableOpLtransformer_block_15_sequential_15_dense_139_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp?
4transformer_block_15/sequential_15/dense_139/BiasAddBiasAdd?transformer_block_15/sequential_15/dense_139/Tensordot:output:0Ktransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 26
4transformer_block_15/sequential_15/dense_139/BiasAdd?
1transformer_block_15/sequential_15/dense_139/ReluRelu=transformer_block_15/sequential_15/dense_139/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 23
1transformer_block_15/sequential_15/dense_139/Relu?
Etransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOpReadVariableOpNtransformer_block_15_sequential_15_dense_140_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02G
Etransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp?
;transformer_block_15/sequential_15/dense_140/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block_15/sequential_15/dense_140/Tensordot/axes?
;transformer_block_15/sequential_15/dense_140/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2=
;transformer_block_15/sequential_15/dense_140/Tensordot/free?
<transformer_block_15/sequential_15/dense_140/Tensordot/ShapeShape?transformer_block_15/sequential_15/dense_139/Relu:activations:0*
T0*
_output_shapes
:2>
<transformer_block_15/sequential_15/dense_140/Tensordot/Shape?
Dtransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2/axis?
?transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2GatherV2Etransformer_block_15/sequential_15/dense_140/Tensordot/Shape:output:0Dtransformer_block_15/sequential_15/dense_140/Tensordot/free:output:0Mtransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2A
?transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2?
Ftransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2H
Ftransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1/axis?
Atransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1GatherV2Etransformer_block_15/sequential_15/dense_140/Tensordot/Shape:output:0Dtransformer_block_15/sequential_15/dense_140/Tensordot/axes:output:0Otransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2C
Atransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1?
<transformer_block_15/sequential_15/dense_140/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<transformer_block_15/sequential_15/dense_140/Tensordot/Const?
;transformer_block_15/sequential_15/dense_140/Tensordot/ProdProdHtransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2:output:0Etransformer_block_15/sequential_15/dense_140/Tensordot/Const:output:0*
T0*
_output_shapes
: 2=
;transformer_block_15/sequential_15/dense_140/Tensordot/Prod?
>transformer_block_15/sequential_15/dense_140/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>transformer_block_15/sequential_15/dense_140/Tensordot/Const_1?
=transformer_block_15/sequential_15/dense_140/Tensordot/Prod_1ProdJtransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1:output:0Gtransformer_block_15/sequential_15/dense_140/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2?
=transformer_block_15/sequential_15/dense_140/Tensordot/Prod_1?
Btransformer_block_15/sequential_15/dense_140/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_15/sequential_15/dense_140/Tensordot/concat/axis?
=transformer_block_15/sequential_15/dense_140/Tensordot/concatConcatV2Dtransformer_block_15/sequential_15/dense_140/Tensordot/free:output:0Dtransformer_block_15/sequential_15/dense_140/Tensordot/axes:output:0Ktransformer_block_15/sequential_15/dense_140/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_15/sequential_15/dense_140/Tensordot/concat?
<transformer_block_15/sequential_15/dense_140/Tensordot/stackPackDtransformer_block_15/sequential_15/dense_140/Tensordot/Prod:output:0Ftransformer_block_15/sequential_15/dense_140/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_15/sequential_15/dense_140/Tensordot/stack?
@transformer_block_15/sequential_15/dense_140/Tensordot/transpose	Transpose?transformer_block_15/sequential_15/dense_139/Relu:activations:0Ftransformer_block_15/sequential_15/dense_140/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_15/sequential_15/dense_140/Tensordot/transpose?
>transformer_block_15/sequential_15/dense_140/Tensordot/ReshapeReshapeDtransformer_block_15/sequential_15/dense_140/Tensordot/transpose:y:0Etransformer_block_15/sequential_15/dense_140/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2@
>transformer_block_15/sequential_15/dense_140/Tensordot/Reshape?
=transformer_block_15/sequential_15/dense_140/Tensordot/MatMulMatMulGtransformer_block_15/sequential_15/dense_140/Tensordot/Reshape:output:0Mtransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2?
=transformer_block_15/sequential_15/dense_140/Tensordot/MatMul?
>transformer_block_15/sequential_15/dense_140/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2@
>transformer_block_15/sequential_15/dense_140/Tensordot/Const_2?
Dtransformer_block_15/sequential_15/dense_140/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_15/sequential_15/dense_140/Tensordot/concat_1/axis?
?transformer_block_15/sequential_15/dense_140/Tensordot/concat_1ConcatV2Htransformer_block_15/sequential_15/dense_140/Tensordot/GatherV2:output:0Gtransformer_block_15/sequential_15/dense_140/Tensordot/Const_2:output:0Mtransformer_block_15/sequential_15/dense_140/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_15/sequential_15/dense_140/Tensordot/concat_1?
6transformer_block_15/sequential_15/dense_140/TensordotReshapeGtransformer_block_15/sequential_15/dense_140/Tensordot/MatMul:product:0Htransformer_block_15/sequential_15/dense_140/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 28
6transformer_block_15/sequential_15/dense_140/Tensordot?
Ctransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOpReadVariableOpLtransformer_block_15_sequential_15_dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp?
4transformer_block_15/sequential_15/dense_140/BiasAddBiasAdd?transformer_block_15/sequential_15/dense_140/Tensordot:output:0Ktransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 26
4transformer_block_15/sequential_15/dense_140/BiasAdd?
-transformer_block_15/dropout_31/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2/
-transformer_block_15/dropout_31/dropout/Const?
+transformer_block_15/dropout_31/dropout/MulMul=transformer_block_15/sequential_15/dense_140/BiasAdd:output:06transformer_block_15/dropout_31/dropout/Const:output:0*
T0*+
_output_shapes
:?????????( 2-
+transformer_block_15/dropout_31/dropout/Mul?
-transformer_block_15/dropout_31/dropout/ShapeShape=transformer_block_15/sequential_15/dense_140/BiasAdd:output:0*
T0*
_output_shapes
:2/
-transformer_block_15/dropout_31/dropout/Shape?
Dtransformer_block_15/dropout_31/dropout/random_uniform/RandomUniformRandomUniform6transformer_block_15/dropout_31/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????( *
dtype02F
Dtransformer_block_15/dropout_31/dropout/random_uniform/RandomUniform?
6transformer_block_15/dropout_31/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=28
6transformer_block_15/dropout_31/dropout/GreaterEqual/y?
4transformer_block_15/dropout_31/dropout/GreaterEqualGreaterEqualMtransformer_block_15/dropout_31/dropout/random_uniform/RandomUniform:output:0?transformer_block_15/dropout_31/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????( 26
4transformer_block_15/dropout_31/dropout/GreaterEqual?
,transformer_block_15/dropout_31/dropout/CastCast8transformer_block_15/dropout_31/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????( 2.
,transformer_block_15/dropout_31/dropout/Cast?
-transformer_block_15/dropout_31/dropout/Mul_1Mul/transformer_block_15/dropout_31/dropout/Mul:z:00transformer_block_15/dropout_31/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????( 2/
-transformer_block_15/dropout_31/dropout/Mul_1?
transformer_block_15/add_1AddV2?transformer_block_15/layer_normalization_30/batchnorm/add_1:z:01transformer_block_15/dropout_31/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_15/add_1?
Jtransformer_block_15/layer_normalization_31/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jtransformer_block_15/layer_normalization_31/moments/mean/reduction_indices?
8transformer_block_15/layer_normalization_31/moments/meanMeantransformer_block_15/add_1:z:0Stransformer_block_15/layer_normalization_31/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2:
8transformer_block_15/layer_normalization_31/moments/mean?
@transformer_block_15/layer_normalization_31/moments/StopGradientStopGradientAtransformer_block_15/layer_normalization_31/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2B
@transformer_block_15/layer_normalization_31/moments/StopGradient?
Etransformer_block_15/layer_normalization_31/moments/SquaredDifferenceSquaredDifferencetransformer_block_15/add_1:z:0Itransformer_block_15/layer_normalization_31/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2G
Etransformer_block_15/layer_normalization_31/moments/SquaredDifference?
Ntransformer_block_15/layer_normalization_31/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
Ntransformer_block_15/layer_normalization_31/moments/variance/reduction_indices?
<transformer_block_15/layer_normalization_31/moments/varianceMeanItransformer_block_15/layer_normalization_31/moments/SquaredDifference:z:0Wtransformer_block_15/layer_normalization_31/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2>
<transformer_block_15/layer_normalization_31/moments/variance?
;transformer_block_15/layer_normalization_31/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52=
;transformer_block_15/layer_normalization_31/batchnorm/add/y?
9transformer_block_15/layer_normalization_31/batchnorm/addAddV2Etransformer_block_15/layer_normalization_31/moments/variance:output:0Dtransformer_block_15/layer_normalization_31/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2;
9transformer_block_15/layer_normalization_31/batchnorm/add?
;transformer_block_15/layer_normalization_31/batchnorm/RsqrtRsqrt=transformer_block_15/layer_normalization_31/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2=
;transformer_block_15/layer_normalization_31/batchnorm/Rsqrt?
Htransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOpReadVariableOpQtransformer_block_15_layer_normalization_31_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02J
Htransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp?
9transformer_block_15/layer_normalization_31/batchnorm/mulMul?transformer_block_15/layer_normalization_31/batchnorm/Rsqrt:y:0Ptransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_15/layer_normalization_31/batchnorm/mul?
;transformer_block_15/layer_normalization_31/batchnorm/mul_1Multransformer_block_15/add_1:z:0=transformer_block_15/layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_31/batchnorm/mul_1?
;transformer_block_15/layer_normalization_31/batchnorm/mul_2MulAtransformer_block_15/layer_normalization_31/moments/mean:output:0=transformer_block_15/layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_31/batchnorm/mul_2?
Dtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOpReadVariableOpMtransformer_block_15_layer_normalization_31_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02F
Dtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp?
9transformer_block_15/layer_normalization_31/batchnorm/subSubLtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp:value:0?transformer_block_15/layer_normalization_31/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_15/layer_normalization_31/batchnorm/sub?
;transformer_block_15/layer_normalization_31/batchnorm/add_1AddV2?transformer_block_15/layer_normalization_31/batchnorm/mul_1:z:0=transformer_block_15/layer_normalization_31/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2=
;transformer_block_15/layer_normalization_31/batchnorm/add_1?
2global_average_pooling1d_15/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :24
2global_average_pooling1d_15/Mean/reduction_indices?
 global_average_pooling1d_15/MeanMean?transformer_block_15/layer_normalization_31/batchnorm/add_1:z:0;global_average_pooling1d_15/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2"
 global_average_pooling1d_15/Mean?
 aux_output/MatMul/ReadVariableOpReadVariableOp)aux_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 aux_output/MatMul/ReadVariableOp?
aux_output/MatMulMatMul)global_average_pooling1d_15/Mean:output:0(aux_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
aux_output/MatMul?
!aux_output/BiasAdd/ReadVariableOpReadVariableOp*aux_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!aux_output/BiasAdd/ReadVariableOp?
aux_output/BiasAddBiasAddaux_output/MatMul:product:0)aux_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
aux_output/BiasAdd?
aux_output/SigmoidSigmoidaux_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
aux_output/Sigmoidz
concatenate_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_15/concat/axis?
concatenate_15/concatConcatV2aux_output/Sigmoid:y:0inputs_1inputs_2#concatenate_15/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_15/concat?
dense_141/MatMul/ReadVariableOpReadVariableOp(dense_141_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_141/MatMul/ReadVariableOp?
dense_141/MatMulMatMulconcatenate_15/concat:output:0'dense_141/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_141/MatMul?
 dense_141/BiasAdd/ReadVariableOpReadVariableOp)dense_141_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_141/BiasAdd/ReadVariableOp?
dense_141/BiasAddBiasAdddense_141/MatMul:product:0(dense_141/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_141/BiasAddv
dense_141/ReluReludense_141/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_141/Relu?
dense_142/MatMul/ReadVariableOpReadVariableOp(dense_142_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_142/MatMul/ReadVariableOp?
dense_142/MatMulMatMuldense_141/Relu:activations:0'dense_142/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_142/MatMul?
 dense_142/BiasAdd/ReadVariableOpReadVariableOp)dense_142_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_142/BiasAdd/ReadVariableOp?
dense_142/BiasAddBiasAdddense_142/MatMul:product:0(dense_142/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_142/BiasAddv
dense_142/ReluReludense_142/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_142/Relu?
dense_143/MatMul/ReadVariableOpReadVariableOp(dense_143_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02!
dense_143/MatMul/ReadVariableOp?
dense_143/MatMulMatMuldense_142/Relu:activations:0'dense_143/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_143/MatMul?
 dense_143/BiasAdd/ReadVariableOpReadVariableOp)dense_143_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_143/BiasAdd/ReadVariableOp?
dense_143/BiasAddBiasAdddense_143/MatMul:product:0(dense_143/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_143/BiasAddv
dense_143/ReluReludense_143/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_143/Relu?
!main_output/MatMul/ReadVariableOpReadVariableOp*main_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!main_output/MatMul/ReadVariableOp?
main_output/MatMulMatMuldense_143/Relu:activations:0)main_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
main_output/MatMul?
"main_output/BiasAdd/ReadVariableOpReadVariableOp+main_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"main_output/BiasAdd/ReadVariableOp?
main_output/BiasAddBiasAddmain_output/MatMul:product:0*main_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
main_output/BiasAdd?
main_output/SigmoidSigmoidmain_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
main_output/Sigmoidr
IdentityIdentitymain_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityu

Identity_1Identityaux_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp"^aux_output/BiasAdd/ReadVariableOp!^aux_output/MatMul/ReadVariableOp!^dense_141/BiasAdd/ReadVariableOp ^dense_141/MatMul/ReadVariableOp!^dense_142/BiasAdd/ReadVariableOp ^dense_142/MatMul/ReadVariableOp!^dense_143/BiasAdd/ReadVariableOp ^dense_143/MatMul/ReadVariableOp#^main_output/BiasAdd/ReadVariableOp"^main_output/MatMul/ReadVariableOp>^token_and_position_embedding_15/embedding_30/embedding_lookup>^token_and_position_embedding_15/embedding_31/embedding_lookupE^transformer_block_15/layer_normalization_30/batchnorm/ReadVariableOpI^transformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOpE^transformer_block_15/layer_normalization_31/batchnorm/ReadVariableOpI^transformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOpS^transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpU^transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpS^transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpU^transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpS^transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpU^transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpS^transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpU^transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpD^transformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOpF^transformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOpD^transformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOpF^transformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!aux_output/BiasAdd/ReadVariableOp!aux_output/BiasAdd/ReadVariableOp2D
 aux_output/MatMul/ReadVariableOp aux_output/MatMul/ReadVariableOp2D
 dense_141/BiasAdd/ReadVariableOp dense_141/BiasAdd/ReadVariableOp2B
dense_141/MatMul/ReadVariableOpdense_141/MatMul/ReadVariableOp2D
 dense_142/BiasAdd/ReadVariableOp dense_142/BiasAdd/ReadVariableOp2B
dense_142/MatMul/ReadVariableOpdense_142/MatMul/ReadVariableOp2D
 dense_143/BiasAdd/ReadVariableOp dense_143/BiasAdd/ReadVariableOp2B
dense_143/MatMul/ReadVariableOpdense_143/MatMul/ReadVariableOp2H
"main_output/BiasAdd/ReadVariableOp"main_output/BiasAdd/ReadVariableOp2F
!main_output/MatMul/ReadVariableOp!main_output/MatMul/ReadVariableOp2~
=token_and_position_embedding_15/embedding_30/embedding_lookup=token_and_position_embedding_15/embedding_30/embedding_lookup2~
=token_and_position_embedding_15/embedding_31/embedding_lookup=token_and_position_embedding_15/embedding_31/embedding_lookup2?
Dtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOpDtransformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp2?
Htransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOpHtransformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp2?
Dtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOpDtransformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp2?
Htransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOpHtransformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp2?
Rtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpRtransformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp2?
Ttransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp2?
Rtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpRtransformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp2?
Ttransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp2?
Rtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpRtransformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp2?
Ttransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp2?
Rtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpRtransformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp2?
Ttransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpTtransformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp2?
Ctransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOpCtransformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp2?
Etransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOpEtransformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp2?
Ctransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOpCtransformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp2?
Etransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOpEtransformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?
?
+__inference_dense_141_layer_call_fn_6977996

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_141_layer_call_and_return_conditional_losses_69755442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_token_and_position_embedding_15_layer_call_fn_6977343
x
unknown:( 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_token_and_position_embedding_15_layer_call_and_return_conditional_losses_69752152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????(

_user_specified_namex
?
?
*__inference_model_15_layer_call_fn_6977245
inputs_0
inputs_1
inputs_2
unknown:( 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18:

unknown_19:@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_15_layer_call_and_return_conditional_losses_69756032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2
?
t
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_6977931

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????( :S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
/__inference_sequential_15_layer_call_fn_6978183

inputs
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_15_layer_call_and_return_conditional_losses_69750442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
\__inference_token_and_position_embedding_15_layer_call_and_return_conditional_losses_6977334
x7
%embedding_31_embedding_lookup_6977321:( 7
%embedding_30_embedding_lookup_6977327: 
identity??embedding_30/embedding_lookup?embedding_31/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta?
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:?????????2
range?
embedding_31/embedding_lookupResourceGather%embedding_31_embedding_lookup_6977321range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_31/embedding_lookup/6977321*'
_output_shapes
:????????? *
dtype02
embedding_31/embedding_lookup?
&embedding_31/embedding_lookup/IdentityIdentity&embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_31/embedding_lookup/6977321*'
_output_shapes
:????????? 2(
&embedding_31/embedding_lookup/Identity?
(embedding_31/embedding_lookup/Identity_1Identity/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2*
(embedding_31/embedding_lookup/Identity_1r
embedding_30/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:?????????(2
embedding_30/Cast?
embedding_30/embedding_lookupResourceGather%embedding_30_embedding_lookup_6977327embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_30/embedding_lookup/6977327*+
_output_shapes
:?????????( *
dtype02
embedding_30/embedding_lookup?
&embedding_30/embedding_lookup/IdentityIdentity&embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_30/embedding_lookup/6977327*+
_output_shapes
:?????????( 2(
&embedding_30/embedding_lookup/Identity?
(embedding_30/embedding_lookup/Identity_1Identity/embedding_30/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2*
(embedding_30/embedding_lookup/Identity_1?
addAddV21embedding_30/embedding_lookup/Identity_1:output:01embedding_31/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2
addf
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp^embedding_30/embedding_lookup^embedding_31/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 2>
embedding_30/embedding_lookupembedding_30/embedding_lookup2>
embedding_31/embedding_lookupembedding_31/embedding_lookup:J F
'
_output_shapes
:?????????(

_user_specified_namex
?>
?
E__inference_model_15_layer_call_and_return_conditional_losses_6976409
input_16
	aux_input
aaindex_input9
'token_and_position_embedding_15_6976342:( 9
'token_and_position_embedding_15_6976344: .
transformer_block_15_6976347:  *
transformer_block_15_6976349: .
transformer_block_15_6976351:  *
transformer_block_15_6976353: .
transformer_block_15_6976355:  *
transformer_block_15_6976357: .
transformer_block_15_6976359:  *
transformer_block_15_6976361: *
transformer_block_15_6976363: *
transformer_block_15_6976365: .
transformer_block_15_6976367:  *
transformer_block_15_6976369: .
transformer_block_15_6976371:  *
transformer_block_15_6976373: *
transformer_block_15_6976375: *
transformer_block_15_6976377: $
aux_output_6976381:  
aux_output_6976383:#
dense_141_6976387:@
dense_141_6976389:@#
dense_142_6976392:@@
dense_142_6976394:@#
dense_143_6976397:@@
dense_143_6976399:@%
main_output_6976402:@!
main_output_6976404:
identity

identity_1??"aux_output/StatefulPartitionedCall?!dense_141/StatefulPartitionedCall?!dense_142/StatefulPartitionedCall?!dense_143/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?7token_and_position_embedding_15/StatefulPartitionedCall?,transformer_block_15/StatefulPartitionedCall?
7token_and_position_embedding_15/StatefulPartitionedCallStatefulPartitionedCallinput_16'token_and_position_embedding_15_6976342'token_and_position_embedding_15_6976344*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_token_and_position_embedding_15_layer_call_and_return_conditional_losses_697521529
7token_and_position_embedding_15/StatefulPartitionedCall?
,transformer_block_15/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_15/StatefulPartitionedCall:output:0transformer_block_15_6976347transformer_block_15_6976349transformer_block_15_6976351transformer_block_15_6976353transformer_block_15_6976355transformer_block_15_6976357transformer_block_15_6976359transformer_block_15_6976361transformer_block_15_6976363transformer_block_15_6976365transformer_block_15_6976367transformer_block_15_6976369transformer_block_15_6976371transformer_block_15_6976373transformer_block_15_6976375transformer_block_15_6976377*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_69754652.
,transformer_block_15/StatefulPartitionedCall?
+global_average_pooling1d_15/PartitionedCallPartitionedCall5transformer_block_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *a
f\RZ
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_69755042-
+global_average_pooling1d_15/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_15/PartitionedCall:output:0aux_output_6976381aux_output_6976383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_aux_output_layer_call_and_return_conditional_losses_69755172$
"aux_output/StatefulPartitionedCall?
concatenate_15/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0	aux_inputaaindex_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_15_layer_call_and_return_conditional_losses_69755312 
concatenate_15/PartitionedCall?
!dense_141/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0dense_141_6976387dense_141_6976389*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_141_layer_call_and_return_conditional_losses_69755442#
!dense_141/StatefulPartitionedCall?
!dense_142/StatefulPartitionedCallStatefulPartitionedCall*dense_141/StatefulPartitionedCall:output:0dense_142_6976392dense_142_6976394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_142_layer_call_and_return_conditional_losses_69755612#
!dense_142/StatefulPartitionedCall?
!dense_143/StatefulPartitionedCallStatefulPartitionedCall*dense_142/StatefulPartitionedCall:output:0dense_143_6976397dense_143_6976399*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_143_layer_call_and_return_conditional_losses_69755782#
!dense_143/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall*dense_143/StatefulPartitionedCall:output:0main_output_6976402main_output_6976404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_main_output_layer_call_and_return_conditional_losses_69755952%
#main_output/StatefulPartitionedCall?
IdentityIdentity,main_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity+aux_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp#^aux_output/StatefulPartitionedCall"^dense_141/StatefulPartitionedCall"^dense_142/StatefulPartitionedCall"^dense_143/StatefulPartitionedCall$^main_output/StatefulPartitionedCall8^token_and_position_embedding_15/StatefulPartitionedCall-^transformer_block_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2F
!dense_141/StatefulPartitionedCall!dense_141/StatefulPartitionedCall2F
!dense_142/StatefulPartitionedCall!dense_142/StatefulPartitionedCall2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2r
7token_and_position_embedding_15/StatefulPartitionedCall7token_and_position_embedding_15/StatefulPartitionedCall2\
,transformer_block_15/StatefulPartitionedCall,transformer_block_15/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
input_16:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input:VR
'
_output_shapes
:?????????
'
_user_specified_nameaaindex_input
?
?
,__inference_aux_output_layer_call_fn_6977961

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_aux_output_layer_call_and_return_conditional_losses_69755172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
/__inference_sequential_15_layer_call_fn_6978196

inputs
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_15_layer_call_and_return_conditional_losses_69751042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
??
?
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_6976025

inputsZ
Hmulti_head_self_attention_15_dense_135_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_135_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_136_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_136_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_137_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_137_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_138_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_138_biasadd_readvariableop_resource: J
<layer_normalization_30_batchnorm_mul_readvariableop_resource: F
8layer_normalization_30_batchnorm_readvariableop_resource: K
9sequential_15_dense_139_tensordot_readvariableop_resource:  E
7sequential_15_dense_139_biasadd_readvariableop_resource: K
9sequential_15_dense_140_tensordot_readvariableop_resource:  E
7sequential_15_dense_140_biasadd_readvariableop_resource: J
<layer_normalization_31_batchnorm_mul_readvariableop_resource: F
8layer_normalization_31_batchnorm_readvariableop_resource: 
identity??/layer_normalization_30/batchnorm/ReadVariableOp?3layer_normalization_30/batchnorm/mul/ReadVariableOp?/layer_normalization_31/batchnorm/ReadVariableOp?3layer_normalization_31/batchnorm/mul/ReadVariableOp?=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?.sequential_15/dense_139/BiasAdd/ReadVariableOp?0sequential_15/dense_139/Tensordot/ReadVariableOp?.sequential_15/dense_140/BiasAdd/ReadVariableOp?0sequential_15/dense_140/Tensordot/ReadVariableOp~
"multi_head_self_attention_15/ShapeShapeinputs*
T0*
_output_shapes
:2$
"multi_head_self_attention_15/Shape?
0multi_head_self_attention_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0multi_head_self_attention_15/strided_slice/stack?
2multi_head_self_attention_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2multi_head_self_attention_15/strided_slice/stack_1?
2multi_head_self_attention_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2multi_head_self_attention_15/strided_slice/stack_2?
*multi_head_self_attention_15/strided_sliceStridedSlice+multi_head_self_attention_15/Shape:output:09multi_head_self_attention_15/strided_slice/stack:output:0;multi_head_self_attention_15/strided_slice/stack_1:output:0;multi_head_self_attention_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*multi_head_self_attention_15/strided_slice?
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_135_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_135/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_135/Tensordot/axes?
5multi_head_self_attention_15/dense_135/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_135/Tensordot/free?
6multi_head_self_attention_15/dense_135/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_135/Tensordot/Shape?
>multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_135/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_135/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_135/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_135/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_135/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_135/Tensordot/Const?
5multi_head_self_attention_15/dense_135/Tensordot/ProdProdBmulti_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_135/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_135/Tensordot/Prod?
8multi_head_self_attention_15/dense_135/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_135/Tensordot/Const_1?
7multi_head_self_attention_15/dense_135/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_135/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_135/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_135/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_135/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_135/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_135/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_135/Tensordot/free:output:0>multi_head_self_attention_15/dense_135/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_135/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_135/Tensordot/concat?
6multi_head_self_attention_15/dense_135/Tensordot/stackPack>multi_head_self_attention_15/dense_135/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_135/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_135/Tensordot/stack?
:multi_head_self_attention_15/dense_135/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_135/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_135/Tensordot/transpose?
8multi_head_self_attention_15/dense_135/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_135/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_135/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_135/Tensordot/Reshape?
7multi_head_self_attention_15/dense_135/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_135/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_135/Tensordot/MatMul?
8multi_head_self_attention_15/dense_135/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_135/Tensordot/Const_2?
>multi_head_self_attention_15/dense_135/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_135/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_135/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_135/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_135/Tensordot/concat_1?
0multi_head_self_attention_15/dense_135/TensordotReshapeAmulti_head_self_attention_15/dense_135/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_135/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_135/Tensordot?
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_135_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_135/BiasAddBiasAdd9multi_head_self_attention_15/dense_135/Tensordot:output:0Emulti_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_135/BiasAdd?
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_136_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_136/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_136/Tensordot/axes?
5multi_head_self_attention_15/dense_136/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_136/Tensordot/free?
6multi_head_self_attention_15/dense_136/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_136/Tensordot/Shape?
>multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_136/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_136/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_136/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_136/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_136/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_136/Tensordot/Const?
5multi_head_self_attention_15/dense_136/Tensordot/ProdProdBmulti_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_136/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_136/Tensordot/Prod?
8multi_head_self_attention_15/dense_136/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_136/Tensordot/Const_1?
7multi_head_self_attention_15/dense_136/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_136/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_136/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_136/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_136/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_136/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_136/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_136/Tensordot/free:output:0>multi_head_self_attention_15/dense_136/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_136/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_136/Tensordot/concat?
6multi_head_self_attention_15/dense_136/Tensordot/stackPack>multi_head_self_attention_15/dense_136/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_136/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_136/Tensordot/stack?
:multi_head_self_attention_15/dense_136/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_136/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_136/Tensordot/transpose?
8multi_head_self_attention_15/dense_136/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_136/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_136/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_136/Tensordot/Reshape?
7multi_head_self_attention_15/dense_136/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_136/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_136/Tensordot/MatMul?
8multi_head_self_attention_15/dense_136/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_136/Tensordot/Const_2?
>multi_head_self_attention_15/dense_136/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_136/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_136/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_136/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_136/Tensordot/concat_1?
0multi_head_self_attention_15/dense_136/TensordotReshapeAmulti_head_self_attention_15/dense_136/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_136/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_136/Tensordot?
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_136_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_136/BiasAddBiasAdd9multi_head_self_attention_15/dense_136/Tensordot:output:0Emulti_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_136/BiasAdd?
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_137_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_137/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_137/Tensordot/axes?
5multi_head_self_attention_15/dense_137/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_137/Tensordot/free?
6multi_head_self_attention_15/dense_137/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_137/Tensordot/Shape?
>multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_137/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_137/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_137/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_137/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_137/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_137/Tensordot/Const?
5multi_head_self_attention_15/dense_137/Tensordot/ProdProdBmulti_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_137/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_137/Tensordot/Prod?
8multi_head_self_attention_15/dense_137/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_137/Tensordot/Const_1?
7multi_head_self_attention_15/dense_137/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_137/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_137/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_137/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_137/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_137/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_137/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_137/Tensordot/free:output:0>multi_head_self_attention_15/dense_137/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_137/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_137/Tensordot/concat?
6multi_head_self_attention_15/dense_137/Tensordot/stackPack>multi_head_self_attention_15/dense_137/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_137/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_137/Tensordot/stack?
:multi_head_self_attention_15/dense_137/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_137/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_137/Tensordot/transpose?
8multi_head_self_attention_15/dense_137/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_137/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_137/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_137/Tensordot/Reshape?
7multi_head_self_attention_15/dense_137/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_137/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_137/Tensordot/MatMul?
8multi_head_self_attention_15/dense_137/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_137/Tensordot/Const_2?
>multi_head_self_attention_15/dense_137/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_137/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_137/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_137/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_137/Tensordot/concat_1?
0multi_head_self_attention_15/dense_137/TensordotReshapeAmulti_head_self_attention_15/dense_137/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_137/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_137/Tensordot?
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_137_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_137/BiasAddBiasAdd9multi_head_self_attention_15/dense_137/Tensordot:output:0Emulti_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_137/BiasAdd?
,multi_head_self_attention_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,multi_head_self_attention_15/Reshape/shape/1?
,multi_head_self_attention_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2.
,multi_head_self_attention_15/Reshape/shape/2?
,multi_head_self_attention_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2.
,multi_head_self_attention_15/Reshape/shape/3?
*multi_head_self_attention_15/Reshape/shapePack3multi_head_self_attention_15/strided_slice:output:05multi_head_self_attention_15/Reshape/shape/1:output:05multi_head_self_attention_15/Reshape/shape/2:output:05multi_head_self_attention_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2,
*multi_head_self_attention_15/Reshape/shape?
$multi_head_self_attention_15/ReshapeReshape7multi_head_self_attention_15/dense_135/BiasAdd:output:03multi_head_self_attention_15/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_15/Reshape?
+multi_head_self_attention_15/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+multi_head_self_attention_15/transpose/perm?
&multi_head_self_attention_15/transpose	Transpose-multi_head_self_attention_15/Reshape:output:04multi_head_self_attention_15/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/transpose?
.multi_head_self_attention_15/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_1/shape/1?
.multi_head_self_attention_15/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_1/shape/2?
.multi_head_self_attention_15/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_1/shape/3?
,multi_head_self_attention_15/Reshape_1/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_1/shape/1:output:07multi_head_self_attention_15/Reshape_1/shape/2:output:07multi_head_self_attention_15/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_1/shape?
&multi_head_self_attention_15/Reshape_1Reshape7multi_head_self_attention_15/dense_136/BiasAdd:output:05multi_head_self_attention_15/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/Reshape_1?
-multi_head_self_attention_15/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_1/perm?
(multi_head_self_attention_15/transpose_1	Transpose/multi_head_self_attention_15/Reshape_1:output:06multi_head_self_attention_15/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_1?
.multi_head_self_attention_15/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_2/shape/1?
.multi_head_self_attention_15/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_2/shape/2?
.multi_head_self_attention_15/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_2/shape/3?
,multi_head_self_attention_15/Reshape_2/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_2/shape/1:output:07multi_head_self_attention_15/Reshape_2/shape/2:output:07multi_head_self_attention_15/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_2/shape?
&multi_head_self_attention_15/Reshape_2Reshape7multi_head_self_attention_15/dense_137/BiasAdd:output:05multi_head_self_attention_15/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/Reshape_2?
-multi_head_self_attention_15/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_2/perm?
(multi_head_self_attention_15/transpose_2	Transpose/multi_head_self_attention_15/Reshape_2:output:06multi_head_self_attention_15/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_2?
#multi_head_self_attention_15/MatMulBatchMatMulV2*multi_head_self_attention_15/transpose:y:0,multi_head_self_attention_15/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2%
#multi_head_self_attention_15/MatMul?
$multi_head_self_attention_15/Shape_1Shape,multi_head_self_attention_15/transpose_1:y:0*
T0*
_output_shapes
:2&
$multi_head_self_attention_15/Shape_1?
2multi_head_self_attention_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2multi_head_self_attention_15/strided_slice_1/stack?
4multi_head_self_attention_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_15/strided_slice_1/stack_1?
4multi_head_self_attention_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4multi_head_self_attention_15/strided_slice_1/stack_2?
,multi_head_self_attention_15/strided_slice_1StridedSlice-multi_head_self_attention_15/Shape_1:output:0;multi_head_self_attention_15/strided_slice_1/stack:output:0=multi_head_self_attention_15/strided_slice_1/stack_1:output:0=multi_head_self_attention_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,multi_head_self_attention_15/strided_slice_1?
!multi_head_self_attention_15/CastCast5multi_head_self_attention_15/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!multi_head_self_attention_15/Cast?
!multi_head_self_attention_15/SqrtSqrt%multi_head_self_attention_15/Cast:y:0*
T0*
_output_shapes
: 2#
!multi_head_self_attention_15/Sqrt?
$multi_head_self_attention_15/truedivRealDiv,multi_head_self_attention_15/MatMul:output:0%multi_head_self_attention_15/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2&
$multi_head_self_attention_15/truediv?
$multi_head_self_attention_15/SoftmaxSoftmax(multi_head_self_attention_15/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2&
$multi_head_self_attention_15/Softmax?
%multi_head_self_attention_15/MatMul_1BatchMatMulV2.multi_head_self_attention_15/Softmax:softmax:0,multi_head_self_attention_15/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_15/MatMul_1?
-multi_head_self_attention_15/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_3/perm?
(multi_head_self_attention_15/transpose_3	Transpose.multi_head_self_attention_15/MatMul_1:output:06multi_head_self_attention_15/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_3?
.multi_head_self_attention_15/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_3/shape/1?
.multi_head_self_attention_15/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_self_attention_15/Reshape_3/shape/2?
,multi_head_self_attention_15/Reshape_3/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_3/shape/1:output:07multi_head_self_attention_15/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_3/shape?
&multi_head_self_attention_15/Reshape_3Reshape,multi_head_self_attention_15/transpose_3:y:05multi_head_self_attention_15/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2(
&multi_head_self_attention_15/Reshape_3?
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_138_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_138/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_138/Tensordot/axes?
5multi_head_self_attention_15/dense_138/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_138/Tensordot/free?
6multi_head_self_attention_15/dense_138/Tensordot/ShapeShape/multi_head_self_attention_15/Reshape_3:output:0*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_138/Tensordot/Shape?
>multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_138/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_138/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_138/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_138/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_138/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_138/Tensordot/Const?
5multi_head_self_attention_15/dense_138/Tensordot/ProdProdBmulti_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_138/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_138/Tensordot/Prod?
8multi_head_self_attention_15/dense_138/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_138/Tensordot/Const_1?
7multi_head_self_attention_15/dense_138/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_138/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_138/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_138/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_138/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_138/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_138/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_138/Tensordot/free:output:0>multi_head_self_attention_15/dense_138/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_138/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_138/Tensordot/concat?
6multi_head_self_attention_15/dense_138/Tensordot/stackPack>multi_head_self_attention_15/dense_138/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_138/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_138/Tensordot/stack?
:multi_head_self_attention_15/dense_138/Tensordot/transpose	Transpose/multi_head_self_attention_15/Reshape_3:output:0@multi_head_self_attention_15/dense_138/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2<
:multi_head_self_attention_15/dense_138/Tensordot/transpose?
8multi_head_self_attention_15/dense_138/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_138/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_138/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_138/Tensordot/Reshape?
7multi_head_self_attention_15/dense_138/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_138/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_138/Tensordot/MatMul?
8multi_head_self_attention_15/dense_138/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_138/Tensordot/Const_2?
>multi_head_self_attention_15/dense_138/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_138/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_138/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_138/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_138/Tensordot/concat_1?
0multi_head_self_attention_15/dense_138/TensordotReshapeAmulti_head_self_attention_15/dense_138/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_138/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 22
0multi_head_self_attention_15/dense_138/Tensordot?
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_138/BiasAddBiasAdd9multi_head_self_attention_15/dense_138/Tensordot:output:0Emulti_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_15/dense_138/BiasAddy
dropout_30/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_30/dropout/Const?
dropout_30/dropout/MulMul7multi_head_self_attention_15/dense_138/BiasAdd:output:0!dropout_30/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_30/dropout/Mul?
dropout_30/dropout/ShapeShape7multi_head_self_attention_15/dense_138/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_30/dropout/Shape?
/dropout_30/dropout/random_uniform/RandomUniformRandomUniform!dropout_30/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype021
/dropout_30/dropout/random_uniform/RandomUniform?
!dropout_30/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_30/dropout/GreaterEqual/y?
dropout_30/dropout/GreaterEqualGreaterEqual8dropout_30/dropout/random_uniform/RandomUniform:output:0*dropout_30/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2!
dropout_30/dropout/GreaterEqual?
dropout_30/dropout/CastCast#dropout_30/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout_30/dropout/Cast?
dropout_30/dropout/Mul_1Muldropout_30/dropout/Mul:z:0dropout_30/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_30/dropout/Mul_1o
addAddV2inputsdropout_30/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add?
5layer_normalization_30/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_30/moments/mean/reduction_indices?
#layer_normalization_30/moments/meanMeanadd:z:0>layer_normalization_30/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_30/moments/mean?
+layer_normalization_30/moments/StopGradientStopGradient,layer_normalization_30/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_30/moments/StopGradient?
0layer_normalization_30/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_30/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_30/moments/SquaredDifference?
9layer_normalization_30/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_30/moments/variance/reduction_indices?
'layer_normalization_30/moments/varianceMean4layer_normalization_30/moments/SquaredDifference:z:0Blayer_normalization_30/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_30/moments/variance?
&layer_normalization_30/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_30/batchnorm/add/y?
$layer_normalization_30/batchnorm/addAddV20layer_normalization_30/moments/variance:output:0/layer_normalization_30/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_30/batchnorm/add?
&layer_normalization_30/batchnorm/RsqrtRsqrt(layer_normalization_30/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_30/batchnorm/Rsqrt?
3layer_normalization_30/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_30_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_30/batchnorm/mul/ReadVariableOp?
$layer_normalization_30/batchnorm/mulMul*layer_normalization_30/batchnorm/Rsqrt:y:0;layer_normalization_30/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_30/batchnorm/mul?
&layer_normalization_30/batchnorm/mul_1Muladd:z:0(layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/mul_1?
&layer_normalization_30/batchnorm/mul_2Mul,layer_normalization_30/moments/mean:output:0(layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/mul_2?
/layer_normalization_30/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_30_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_30/batchnorm/ReadVariableOp?
$layer_normalization_30/batchnorm/subSub7layer_normalization_30/batchnorm/ReadVariableOp:value:0*layer_normalization_30/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_30/batchnorm/sub?
&layer_normalization_30/batchnorm/add_1AddV2*layer_normalization_30/batchnorm/mul_1:z:0(layer_normalization_30/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/add_1?
0sequential_15/dense_139/Tensordot/ReadVariableOpReadVariableOp9sequential_15_dense_139_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype022
0sequential_15/dense_139/Tensordot/ReadVariableOp?
&sequential_15/dense_139/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_15/dense_139/Tensordot/axes?
&sequential_15/dense_139/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&sequential_15/dense_139/Tensordot/free?
'sequential_15/dense_139/Tensordot/ShapeShape*layer_normalization_30/batchnorm/add_1:z:0*
T0*
_output_shapes
:2)
'sequential_15/dense_139/Tensordot/Shape?
/sequential_15/dense_139/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_139/Tensordot/GatherV2/axis?
*sequential_15/dense_139/Tensordot/GatherV2GatherV20sequential_15/dense_139/Tensordot/Shape:output:0/sequential_15/dense_139/Tensordot/free:output:08sequential_15/dense_139/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_15/dense_139/Tensordot/GatherV2?
1sequential_15/dense_139/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_15/dense_139/Tensordot/GatherV2_1/axis?
,sequential_15/dense_139/Tensordot/GatherV2_1GatherV20sequential_15/dense_139/Tensordot/Shape:output:0/sequential_15/dense_139/Tensordot/axes:output:0:sequential_15/dense_139/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,sequential_15/dense_139/Tensordot/GatherV2_1?
'sequential_15/dense_139/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_15/dense_139/Tensordot/Const?
&sequential_15/dense_139/Tensordot/ProdProd3sequential_15/dense_139/Tensordot/GatherV2:output:00sequential_15/dense_139/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&sequential_15/dense_139/Tensordot/Prod?
)sequential_15/dense_139/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_139/Tensordot/Const_1?
(sequential_15/dense_139/Tensordot/Prod_1Prod5sequential_15/dense_139/Tensordot/GatherV2_1:output:02sequential_15/dense_139/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(sequential_15/dense_139/Tensordot/Prod_1?
-sequential_15/dense_139/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_15/dense_139/Tensordot/concat/axis?
(sequential_15/dense_139/Tensordot/concatConcatV2/sequential_15/dense_139/Tensordot/free:output:0/sequential_15/dense_139/Tensordot/axes:output:06sequential_15/dense_139/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_15/dense_139/Tensordot/concat?
'sequential_15/dense_139/Tensordot/stackPack/sequential_15/dense_139/Tensordot/Prod:output:01sequential_15/dense_139/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_15/dense_139/Tensordot/stack?
+sequential_15/dense_139/Tensordot/transpose	Transpose*layer_normalization_30/batchnorm/add_1:z:01sequential_15/dense_139/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2-
+sequential_15/dense_139/Tensordot/transpose?
)sequential_15/dense_139/Tensordot/ReshapeReshape/sequential_15/dense_139/Tensordot/transpose:y:00sequential_15/dense_139/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)sequential_15/dense_139/Tensordot/Reshape?
(sequential_15/dense_139/Tensordot/MatMulMatMul2sequential_15/dense_139/Tensordot/Reshape:output:08sequential_15/dense_139/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(sequential_15/dense_139/Tensordot/MatMul?
)sequential_15/dense_139/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_139/Tensordot/Const_2?
/sequential_15/dense_139/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_139/Tensordot/concat_1/axis?
*sequential_15/dense_139/Tensordot/concat_1ConcatV23sequential_15/dense_139/Tensordot/GatherV2:output:02sequential_15/dense_139/Tensordot/Const_2:output:08sequential_15/dense_139/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*sequential_15/dense_139/Tensordot/concat_1?
!sequential_15/dense_139/TensordotReshape2sequential_15/dense_139/Tensordot/MatMul:product:03sequential_15/dense_139/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2#
!sequential_15/dense_139/Tensordot?
.sequential_15/dense_139/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_dense_139_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/dense_139/BiasAdd/ReadVariableOp?
sequential_15/dense_139/BiasAddBiasAdd*sequential_15/dense_139/Tensordot:output:06sequential_15/dense_139/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2!
sequential_15/dense_139/BiasAdd?
sequential_15/dense_139/ReluRelu(sequential_15/dense_139/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_15/dense_139/Relu?
0sequential_15/dense_140/Tensordot/ReadVariableOpReadVariableOp9sequential_15_dense_140_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype022
0sequential_15/dense_140/Tensordot/ReadVariableOp?
&sequential_15/dense_140/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_15/dense_140/Tensordot/axes?
&sequential_15/dense_140/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&sequential_15/dense_140/Tensordot/free?
'sequential_15/dense_140/Tensordot/ShapeShape*sequential_15/dense_139/Relu:activations:0*
T0*
_output_shapes
:2)
'sequential_15/dense_140/Tensordot/Shape?
/sequential_15/dense_140/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_140/Tensordot/GatherV2/axis?
*sequential_15/dense_140/Tensordot/GatherV2GatherV20sequential_15/dense_140/Tensordot/Shape:output:0/sequential_15/dense_140/Tensordot/free:output:08sequential_15/dense_140/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_15/dense_140/Tensordot/GatherV2?
1sequential_15/dense_140/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_15/dense_140/Tensordot/GatherV2_1/axis?
,sequential_15/dense_140/Tensordot/GatherV2_1GatherV20sequential_15/dense_140/Tensordot/Shape:output:0/sequential_15/dense_140/Tensordot/axes:output:0:sequential_15/dense_140/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,sequential_15/dense_140/Tensordot/GatherV2_1?
'sequential_15/dense_140/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_15/dense_140/Tensordot/Const?
&sequential_15/dense_140/Tensordot/ProdProd3sequential_15/dense_140/Tensordot/GatherV2:output:00sequential_15/dense_140/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&sequential_15/dense_140/Tensordot/Prod?
)sequential_15/dense_140/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_140/Tensordot/Const_1?
(sequential_15/dense_140/Tensordot/Prod_1Prod5sequential_15/dense_140/Tensordot/GatherV2_1:output:02sequential_15/dense_140/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(sequential_15/dense_140/Tensordot/Prod_1?
-sequential_15/dense_140/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_15/dense_140/Tensordot/concat/axis?
(sequential_15/dense_140/Tensordot/concatConcatV2/sequential_15/dense_140/Tensordot/free:output:0/sequential_15/dense_140/Tensordot/axes:output:06sequential_15/dense_140/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_15/dense_140/Tensordot/concat?
'sequential_15/dense_140/Tensordot/stackPack/sequential_15/dense_140/Tensordot/Prod:output:01sequential_15/dense_140/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_15/dense_140/Tensordot/stack?
+sequential_15/dense_140/Tensordot/transpose	Transpose*sequential_15/dense_139/Relu:activations:01sequential_15/dense_140/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2-
+sequential_15/dense_140/Tensordot/transpose?
)sequential_15/dense_140/Tensordot/ReshapeReshape/sequential_15/dense_140/Tensordot/transpose:y:00sequential_15/dense_140/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)sequential_15/dense_140/Tensordot/Reshape?
(sequential_15/dense_140/Tensordot/MatMulMatMul2sequential_15/dense_140/Tensordot/Reshape:output:08sequential_15/dense_140/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(sequential_15/dense_140/Tensordot/MatMul?
)sequential_15/dense_140/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_140/Tensordot/Const_2?
/sequential_15/dense_140/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_140/Tensordot/concat_1/axis?
*sequential_15/dense_140/Tensordot/concat_1ConcatV23sequential_15/dense_140/Tensordot/GatherV2:output:02sequential_15/dense_140/Tensordot/Const_2:output:08sequential_15/dense_140/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*sequential_15/dense_140/Tensordot/concat_1?
!sequential_15/dense_140/TensordotReshape2sequential_15/dense_140/Tensordot/MatMul:product:03sequential_15/dense_140/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2#
!sequential_15/dense_140/Tensordot?
.sequential_15/dense_140/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/dense_140/BiasAdd/ReadVariableOp?
sequential_15/dense_140/BiasAddBiasAdd*sequential_15/dense_140/Tensordot:output:06sequential_15/dense_140/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2!
sequential_15/dense_140/BiasAddy
dropout_31/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_31/dropout/Const?
dropout_31/dropout/MulMul(sequential_15/dense_140/BiasAdd:output:0!dropout_31/dropout/Const:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_31/dropout/Mul?
dropout_31/dropout/ShapeShape(sequential_15/dense_140/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_31/dropout/Shape?
/dropout_31/dropout/random_uniform/RandomUniformRandomUniform!dropout_31/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????( *
dtype021
/dropout_31/dropout/random_uniform/RandomUniform?
!dropout_31/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_31/dropout/GreaterEqual/y?
dropout_31/dropout/GreaterEqualGreaterEqual8dropout_31/dropout/random_uniform/RandomUniform:output:0*dropout_31/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????( 2!
dropout_31/dropout/GreaterEqual?
dropout_31/dropout/CastCast#dropout_31/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????( 2
dropout_31/dropout/Cast?
dropout_31/dropout/Mul_1Muldropout_31/dropout/Mul:z:0dropout_31/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????( 2
dropout_31/dropout/Mul_1?
add_1AddV2*layer_normalization_30/batchnorm/add_1:z:0dropout_31/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add_1?
5layer_normalization_31/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_31/moments/mean/reduction_indices?
#layer_normalization_31/moments/meanMean	add_1:z:0>layer_normalization_31/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_31/moments/mean?
+layer_normalization_31/moments/StopGradientStopGradient,layer_normalization_31/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_31/moments/StopGradient?
0layer_normalization_31/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_31/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_31/moments/SquaredDifference?
9layer_normalization_31/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_31/moments/variance/reduction_indices?
'layer_normalization_31/moments/varianceMean4layer_normalization_31/moments/SquaredDifference:z:0Blayer_normalization_31/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_31/moments/variance?
&layer_normalization_31/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_31/batchnorm/add/y?
$layer_normalization_31/batchnorm/addAddV20layer_normalization_31/moments/variance:output:0/layer_normalization_31/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_31/batchnorm/add?
&layer_normalization_31/batchnorm/RsqrtRsqrt(layer_normalization_31/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_31/batchnorm/Rsqrt?
3layer_normalization_31/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_31_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_31/batchnorm/mul/ReadVariableOp?
$layer_normalization_31/batchnorm/mulMul*layer_normalization_31/batchnorm/Rsqrt:y:0;layer_normalization_31/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_31/batchnorm/mul?
&layer_normalization_31/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/mul_1?
&layer_normalization_31/batchnorm/mul_2Mul,layer_normalization_31/moments/mean:output:0(layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/mul_2?
/layer_normalization_31/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_31_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_31/batchnorm/ReadVariableOp?
$layer_normalization_31/batchnorm/subSub7layer_normalization_31/batchnorm/ReadVariableOp:value:0*layer_normalization_31/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_31/batchnorm/sub?
&layer_normalization_31/batchnorm/add_1AddV2*layer_normalization_31/batchnorm/mul_1:z:0(layer_normalization_31/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/add_1?
IdentityIdentity*layer_normalization_31/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp0^layer_normalization_30/batchnorm/ReadVariableOp4^layer_normalization_30/batchnorm/mul/ReadVariableOp0^layer_normalization_31/batchnorm/ReadVariableOp4^layer_normalization_31/batchnorm/mul/ReadVariableOp>^multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp/^sequential_15/dense_139/BiasAdd/ReadVariableOp1^sequential_15/dense_139/Tensordot/ReadVariableOp/^sequential_15/dense_140/BiasAdd/ReadVariableOp1^sequential_15/dense_140/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2b
/layer_normalization_30/batchnorm/ReadVariableOp/layer_normalization_30/batchnorm/ReadVariableOp2j
3layer_normalization_30/batchnorm/mul/ReadVariableOp3layer_normalization_30/batchnorm/mul/ReadVariableOp2b
/layer_normalization_31/batchnorm/ReadVariableOp/layer_normalization_31/batchnorm/ReadVariableOp2j
3layer_normalization_31/batchnorm/mul/ReadVariableOp3layer_normalization_31/batchnorm/mul/ReadVariableOp2~
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp2`
.sequential_15/dense_139/BiasAdd/ReadVariableOp.sequential_15/dense_139/BiasAdd/ReadVariableOp2d
0sequential_15/dense_139/Tensordot/ReadVariableOp0sequential_15/dense_139/Tensordot/ReadVariableOp2`
.sequential_15/dense_140/BiasAdd/ReadVariableOp.sequential_15/dense_140/BiasAdd/ReadVariableOp2d
0sequential_15/dense_140/Tensordot/ReadVariableOp0sequential_15/dense_140/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
/__inference_sequential_15_layer_call_fn_6975055
dense_139_input
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_139_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_15_layer_call_and_return_conditional_losses_69750442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????( 
)
_user_specified_namedense_139_input
?
?
F__inference_dense_142_layer_call_and_return_conditional_losses_6975561

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?L
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_6978170

inputs=
+dense_139_tensordot_readvariableop_resource:  7
)dense_139_biasadd_readvariableop_resource: =
+dense_140_tensordot_readvariableop_resource:  7
)dense_140_biasadd_readvariableop_resource: 
identity?? dense_139/BiasAdd/ReadVariableOp?"dense_139/Tensordot/ReadVariableOp? dense_140/BiasAdd/ReadVariableOp?"dense_140/Tensordot/ReadVariableOp?
"dense_139/Tensordot/ReadVariableOpReadVariableOp+dense_139_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02$
"dense_139/Tensordot/ReadVariableOp~
dense_139/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_139/Tensordot/axes?
dense_139/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_139/Tensordot/freel
dense_139/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_139/Tensordot/Shape?
!dense_139/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_139/Tensordot/GatherV2/axis?
dense_139/Tensordot/GatherV2GatherV2"dense_139/Tensordot/Shape:output:0!dense_139/Tensordot/free:output:0*dense_139/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_139/Tensordot/GatherV2?
#dense_139/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_139/Tensordot/GatherV2_1/axis?
dense_139/Tensordot/GatherV2_1GatherV2"dense_139/Tensordot/Shape:output:0!dense_139/Tensordot/axes:output:0,dense_139/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_139/Tensordot/GatherV2_1?
dense_139/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_139/Tensordot/Const?
dense_139/Tensordot/ProdProd%dense_139/Tensordot/GatherV2:output:0"dense_139/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_139/Tensordot/Prod?
dense_139/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_139/Tensordot/Const_1?
dense_139/Tensordot/Prod_1Prod'dense_139/Tensordot/GatherV2_1:output:0$dense_139/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_139/Tensordot/Prod_1?
dense_139/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_139/Tensordot/concat/axis?
dense_139/Tensordot/concatConcatV2!dense_139/Tensordot/free:output:0!dense_139/Tensordot/axes:output:0(dense_139/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_139/Tensordot/concat?
dense_139/Tensordot/stackPack!dense_139/Tensordot/Prod:output:0#dense_139/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_139/Tensordot/stack?
dense_139/Tensordot/transpose	Transposeinputs#dense_139/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_139/Tensordot/transpose?
dense_139/Tensordot/ReshapeReshape!dense_139/Tensordot/transpose:y:0"dense_139/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_139/Tensordot/Reshape?
dense_139/Tensordot/MatMulMatMul$dense_139/Tensordot/Reshape:output:0*dense_139/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_139/Tensordot/MatMul?
dense_139/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_139/Tensordot/Const_2?
!dense_139/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_139/Tensordot/concat_1/axis?
dense_139/Tensordot/concat_1ConcatV2%dense_139/Tensordot/GatherV2:output:0$dense_139/Tensordot/Const_2:output:0*dense_139/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_139/Tensordot/concat_1?
dense_139/TensordotReshape$dense_139/Tensordot/MatMul:product:0%dense_139/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_139/Tensordot?
 dense_139/BiasAdd/ReadVariableOpReadVariableOp)dense_139_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_139/BiasAdd/ReadVariableOp?
dense_139/BiasAddBiasAdddense_139/Tensordot:output:0(dense_139/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_139/BiasAddz
dense_139/ReluReludense_139/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dense_139/Relu?
"dense_140/Tensordot/ReadVariableOpReadVariableOp+dense_140_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02$
"dense_140/Tensordot/ReadVariableOp~
dense_140/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_140/Tensordot/axes?
dense_140/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_140/Tensordot/free?
dense_140/Tensordot/ShapeShapedense_139/Relu:activations:0*
T0*
_output_shapes
:2
dense_140/Tensordot/Shape?
!dense_140/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_140/Tensordot/GatherV2/axis?
dense_140/Tensordot/GatherV2GatherV2"dense_140/Tensordot/Shape:output:0!dense_140/Tensordot/free:output:0*dense_140/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_140/Tensordot/GatherV2?
#dense_140/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_140/Tensordot/GatherV2_1/axis?
dense_140/Tensordot/GatherV2_1GatherV2"dense_140/Tensordot/Shape:output:0!dense_140/Tensordot/axes:output:0,dense_140/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_140/Tensordot/GatherV2_1?
dense_140/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_140/Tensordot/Const?
dense_140/Tensordot/ProdProd%dense_140/Tensordot/GatherV2:output:0"dense_140/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_140/Tensordot/Prod?
dense_140/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_140/Tensordot/Const_1?
dense_140/Tensordot/Prod_1Prod'dense_140/Tensordot/GatherV2_1:output:0$dense_140/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_140/Tensordot/Prod_1?
dense_140/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_140/Tensordot/concat/axis?
dense_140/Tensordot/concatConcatV2!dense_140/Tensordot/free:output:0!dense_140/Tensordot/axes:output:0(dense_140/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_140/Tensordot/concat?
dense_140/Tensordot/stackPack!dense_140/Tensordot/Prod:output:0#dense_140/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_140/Tensordot/stack?
dense_140/Tensordot/transpose	Transposedense_139/Relu:activations:0#dense_140/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_140/Tensordot/transpose?
dense_140/Tensordot/ReshapeReshape!dense_140/Tensordot/transpose:y:0"dense_140/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_140/Tensordot/Reshape?
dense_140/Tensordot/MatMulMatMul$dense_140/Tensordot/Reshape:output:0*dense_140/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_140/Tensordot/MatMul?
dense_140/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_140/Tensordot/Const_2?
!dense_140/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_140/Tensordot/concat_1/axis?
dense_140/Tensordot/concat_1ConcatV2%dense_140/Tensordot/GatherV2:output:0$dense_140/Tensordot/Const_2:output:0*dense_140/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_140/Tensordot/concat_1?
dense_140/TensordotReshape$dense_140/Tensordot/MatMul:product:0%dense_140/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_140/Tensordot?
 dense_140/BiasAdd/ReadVariableOpReadVariableOp)dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_140/BiasAdd/ReadVariableOp?
dense_140/BiasAddBiasAdddense_140/Tensordot:output:0(dense_140/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_140/BiasAddy
IdentityIdentitydense_140/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp!^dense_139/BiasAdd/ReadVariableOp#^dense_139/Tensordot/ReadVariableOp!^dense_140/BiasAdd/ReadVariableOp#^dense_140/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2D
 dense_139/BiasAdd/ReadVariableOp dense_139/BiasAdd/ReadVariableOp2H
"dense_139/Tensordot/ReadVariableOp"dense_139/Tensordot/ReadVariableOp2D
 dense_140/BiasAdd/ReadVariableOp dense_140/BiasAdd/ReadVariableOp2H
"dense_140/Tensordot/ReadVariableOp"dense_140/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?>
?
E__inference_model_15_layer_call_and_return_conditional_losses_6975603

inputs
inputs_1
inputs_29
'token_and_position_embedding_15_6975216:( 9
'token_and_position_embedding_15_6975218: .
transformer_block_15_6975466:  *
transformer_block_15_6975468: .
transformer_block_15_6975470:  *
transformer_block_15_6975472: .
transformer_block_15_6975474:  *
transformer_block_15_6975476: .
transformer_block_15_6975478:  *
transformer_block_15_6975480: *
transformer_block_15_6975482: *
transformer_block_15_6975484: .
transformer_block_15_6975486:  *
transformer_block_15_6975488: .
transformer_block_15_6975490:  *
transformer_block_15_6975492: *
transformer_block_15_6975494: *
transformer_block_15_6975496: $
aux_output_6975518:  
aux_output_6975520:#
dense_141_6975545:@
dense_141_6975547:@#
dense_142_6975562:@@
dense_142_6975564:@#
dense_143_6975579:@@
dense_143_6975581:@%
main_output_6975596:@!
main_output_6975598:
identity

identity_1??"aux_output/StatefulPartitionedCall?!dense_141/StatefulPartitionedCall?!dense_142/StatefulPartitionedCall?!dense_143/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?7token_and_position_embedding_15/StatefulPartitionedCall?,transformer_block_15/StatefulPartitionedCall?
7token_and_position_embedding_15/StatefulPartitionedCallStatefulPartitionedCallinputs'token_and_position_embedding_15_6975216'token_and_position_embedding_15_6975218*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_token_and_position_embedding_15_layer_call_and_return_conditional_losses_697521529
7token_and_position_embedding_15/StatefulPartitionedCall?
,transformer_block_15/StatefulPartitionedCallStatefulPartitionedCall@token_and_position_embedding_15/StatefulPartitionedCall:output:0transformer_block_15_6975466transformer_block_15_6975468transformer_block_15_6975470transformer_block_15_6975472transformer_block_15_6975474transformer_block_15_6975476transformer_block_15_6975478transformer_block_15_6975480transformer_block_15_6975482transformer_block_15_6975484transformer_block_15_6975486transformer_block_15_6975488transformer_block_15_6975490transformer_block_15_6975492transformer_block_15_6975494transformer_block_15_6975496*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_69754652.
,transformer_block_15/StatefulPartitionedCall?
+global_average_pooling1d_15/PartitionedCallPartitionedCall5transformer_block_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *a
f\RZ
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_69755042-
+global_average_pooling1d_15/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall4global_average_pooling1d_15/PartitionedCall:output:0aux_output_6975518aux_output_6975520*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_aux_output_layer_call_and_return_conditional_losses_69755172$
"aux_output/StatefulPartitionedCall?
concatenate_15/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_15_layer_call_and_return_conditional_losses_69755312 
concatenate_15/PartitionedCall?
!dense_141/StatefulPartitionedCallStatefulPartitionedCall'concatenate_15/PartitionedCall:output:0dense_141_6975545dense_141_6975547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_141_layer_call_and_return_conditional_losses_69755442#
!dense_141/StatefulPartitionedCall?
!dense_142/StatefulPartitionedCallStatefulPartitionedCall*dense_141/StatefulPartitionedCall:output:0dense_142_6975562dense_142_6975564*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_142_layer_call_and_return_conditional_losses_69755612#
!dense_142/StatefulPartitionedCall?
!dense_143/StatefulPartitionedCallStatefulPartitionedCall*dense_142/StatefulPartitionedCall:output:0dense_143_6975579dense_143_6975581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_143_layer_call_and_return_conditional_losses_69755782#
!dense_143/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall*dense_143/StatefulPartitionedCall:output:0main_output_6975596main_output_6975598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_main_output_layer_call_and_return_conditional_losses_69755952%
#main_output/StatefulPartitionedCall?
IdentityIdentity,main_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity+aux_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp#^aux_output/StatefulPartitionedCall"^dense_141/StatefulPartitionedCall"^dense_142/StatefulPartitionedCall"^dense_143/StatefulPartitionedCall$^main_output/StatefulPartitionedCall8^token_and_position_embedding_15/StatefulPartitionedCall-^transformer_block_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2F
!dense_141/StatefulPartitionedCall!dense_141/StatefulPartitionedCall2F
!dense_142/StatefulPartitionedCall!dense_142/StatefulPartitionedCall2F
!dense_143/StatefulPartitionedCall!dense_143/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2r
7token_and_position_embedding_15/StatefulPartitionedCall7token_and_position_embedding_15/StatefulPartitionedCall2\
,transformer_block_15/StatefulPartitionedCall,transformer_block_15/StatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
t
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_6977925

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_dense_143_layer_call_and_return_conditional_losses_6978027

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
? 
?
F__inference_dense_140_layer_call_and_return_conditional_losses_6978266

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????( : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
/__inference_sequential_15_layer_call_fn_6975128
dense_139_input
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_139_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_15_layer_call_and_return_conditional_losses_69751042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????( 
)
_user_specified_namedense_139_input
??
?6
 __inference__traced_save_6978598
file_prefix0
,savev2_aux_output_kernel_read_readvariableop.
*savev2_aux_output_bias_read_readvariableop/
+savev2_dense_141_kernel_read_readvariableop-
)savev2_dense_141_bias_read_readvariableop/
+savev2_dense_142_kernel_read_readvariableop-
)savev2_dense_142_bias_read_readvariableop/
+savev2_dense_143_kernel_read_readvariableop-
)savev2_dense_143_bias_read_readvariableop1
-savev2_main_output_kernel_read_readvariableop/
+savev2_main_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopV
Rsavev2_token_and_position_embedding_15_embedding_30_embeddings_read_readvariableopV
Rsavev2_token_and_position_embedding_15_embedding_31_embeddings_read_readvariableopa
]savev2_transformer_block_15_multi_head_self_attention_15_dense_135_kernel_read_readvariableop_
[savev2_transformer_block_15_multi_head_self_attention_15_dense_135_bias_read_readvariableopa
]savev2_transformer_block_15_multi_head_self_attention_15_dense_136_kernel_read_readvariableop_
[savev2_transformer_block_15_multi_head_self_attention_15_dense_136_bias_read_readvariableopa
]savev2_transformer_block_15_multi_head_self_attention_15_dense_137_kernel_read_readvariableop_
[savev2_transformer_block_15_multi_head_self_attention_15_dense_137_bias_read_readvariableopa
]savev2_transformer_block_15_multi_head_self_attention_15_dense_138_kernel_read_readvariableop_
[savev2_transformer_block_15_multi_head_self_attention_15_dense_138_bias_read_readvariableop/
+savev2_dense_139_kernel_read_readvariableop-
)savev2_dense_139_bias_read_readvariableop/
+savev2_dense_140_kernel_read_readvariableop-
)savev2_dense_140_bias_read_readvariableopP
Lsavev2_transformer_block_15_layer_normalization_30_gamma_read_readvariableopO
Ksavev2_transformer_block_15_layer_normalization_30_beta_read_readvariableopP
Lsavev2_transformer_block_15_layer_normalization_31_gamma_read_readvariableopO
Ksavev2_transformer_block_15_layer_normalization_31_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop7
3savev2_adam_aux_output_kernel_m_read_readvariableop5
1savev2_adam_aux_output_bias_m_read_readvariableop6
2savev2_adam_dense_141_kernel_m_read_readvariableop4
0savev2_adam_dense_141_bias_m_read_readvariableop6
2savev2_adam_dense_142_kernel_m_read_readvariableop4
0savev2_adam_dense_142_bias_m_read_readvariableop6
2savev2_adam_dense_143_kernel_m_read_readvariableop4
0savev2_adam_dense_143_bias_m_read_readvariableop8
4savev2_adam_main_output_kernel_m_read_readvariableop6
2savev2_adam_main_output_bias_m_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_15_embedding_30_embeddings_m_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_15_embedding_31_embeddings_m_read_readvariableoph
dsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_135_kernel_m_read_readvariableopf
bsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_135_bias_m_read_readvariableoph
dsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_136_kernel_m_read_readvariableopf
bsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_136_bias_m_read_readvariableoph
dsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_137_kernel_m_read_readvariableopf
bsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_137_bias_m_read_readvariableoph
dsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_138_kernel_m_read_readvariableopf
bsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_138_bias_m_read_readvariableop6
2savev2_adam_dense_139_kernel_m_read_readvariableop4
0savev2_adam_dense_139_bias_m_read_readvariableop6
2savev2_adam_dense_140_kernel_m_read_readvariableop4
0savev2_adam_dense_140_bias_m_read_readvariableopW
Ssavev2_adam_transformer_block_15_layer_normalization_30_gamma_m_read_readvariableopV
Rsavev2_adam_transformer_block_15_layer_normalization_30_beta_m_read_readvariableopW
Ssavev2_adam_transformer_block_15_layer_normalization_31_gamma_m_read_readvariableopV
Rsavev2_adam_transformer_block_15_layer_normalization_31_beta_m_read_readvariableop7
3savev2_adam_aux_output_kernel_v_read_readvariableop5
1savev2_adam_aux_output_bias_v_read_readvariableop6
2savev2_adam_dense_141_kernel_v_read_readvariableop4
0savev2_adam_dense_141_bias_v_read_readvariableop6
2savev2_adam_dense_142_kernel_v_read_readvariableop4
0savev2_adam_dense_142_bias_v_read_readvariableop6
2savev2_adam_dense_143_kernel_v_read_readvariableop4
0savev2_adam_dense_143_bias_v_read_readvariableop8
4savev2_adam_main_output_kernel_v_read_readvariableop6
2savev2_adam_main_output_bias_v_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_15_embedding_30_embeddings_v_read_readvariableop]
Ysavev2_adam_token_and_position_embedding_15_embedding_31_embeddings_v_read_readvariableoph
dsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_135_kernel_v_read_readvariableopf
bsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_135_bias_v_read_readvariableoph
dsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_136_kernel_v_read_readvariableopf
bsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_136_bias_v_read_readvariableoph
dsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_137_kernel_v_read_readvariableopf
bsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_137_bias_v_read_readvariableoph
dsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_138_kernel_v_read_readvariableopf
bsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_138_bias_v_read_readvariableop6
2savev2_adam_dense_139_kernel_v_read_readvariableop4
0savev2_adam_dense_139_bias_v_read_readvariableop6
2savev2_adam_dense_140_kernel_v_read_readvariableop4
0savev2_adam_dense_140_bias_v_read_readvariableopW
Ssavev2_adam_transformer_block_15_layer_normalization_30_gamma_v_read_readvariableopV
Rsavev2_adam_transformer_block_15_layer_normalization_30_beta_v_read_readvariableopW
Ssavev2_adam_transformer_block_15_layer_normalization_31_gamma_v_read_readvariableopV
Rsavev2_adam_transformer_block_15_layer_normalization_31_beta_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?5
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?4
value?4B?4dB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?
value?B?dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?5
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_aux_output_kernel_read_readvariableop*savev2_aux_output_bias_read_readvariableop+savev2_dense_141_kernel_read_readvariableop)savev2_dense_141_bias_read_readvariableop+savev2_dense_142_kernel_read_readvariableop)savev2_dense_142_bias_read_readvariableop+savev2_dense_143_kernel_read_readvariableop)savev2_dense_143_bias_read_readvariableop-savev2_main_output_kernel_read_readvariableop+savev2_main_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopRsavev2_token_and_position_embedding_15_embedding_30_embeddings_read_readvariableopRsavev2_token_and_position_embedding_15_embedding_31_embeddings_read_readvariableop]savev2_transformer_block_15_multi_head_self_attention_15_dense_135_kernel_read_readvariableop[savev2_transformer_block_15_multi_head_self_attention_15_dense_135_bias_read_readvariableop]savev2_transformer_block_15_multi_head_self_attention_15_dense_136_kernel_read_readvariableop[savev2_transformer_block_15_multi_head_self_attention_15_dense_136_bias_read_readvariableop]savev2_transformer_block_15_multi_head_self_attention_15_dense_137_kernel_read_readvariableop[savev2_transformer_block_15_multi_head_self_attention_15_dense_137_bias_read_readvariableop]savev2_transformer_block_15_multi_head_self_attention_15_dense_138_kernel_read_readvariableop[savev2_transformer_block_15_multi_head_self_attention_15_dense_138_bias_read_readvariableop+savev2_dense_139_kernel_read_readvariableop)savev2_dense_139_bias_read_readvariableop+savev2_dense_140_kernel_read_readvariableop)savev2_dense_140_bias_read_readvariableopLsavev2_transformer_block_15_layer_normalization_30_gamma_read_readvariableopKsavev2_transformer_block_15_layer_normalization_30_beta_read_readvariableopLsavev2_transformer_block_15_layer_normalization_31_gamma_read_readvariableopKsavev2_transformer_block_15_layer_normalization_31_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop3savev2_adam_aux_output_kernel_m_read_readvariableop1savev2_adam_aux_output_bias_m_read_readvariableop2savev2_adam_dense_141_kernel_m_read_readvariableop0savev2_adam_dense_141_bias_m_read_readvariableop2savev2_adam_dense_142_kernel_m_read_readvariableop0savev2_adam_dense_142_bias_m_read_readvariableop2savev2_adam_dense_143_kernel_m_read_readvariableop0savev2_adam_dense_143_bias_m_read_readvariableop4savev2_adam_main_output_kernel_m_read_readvariableop2savev2_adam_main_output_bias_m_read_readvariableopYsavev2_adam_token_and_position_embedding_15_embedding_30_embeddings_m_read_readvariableopYsavev2_adam_token_and_position_embedding_15_embedding_31_embeddings_m_read_readvariableopdsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_135_kernel_m_read_readvariableopbsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_135_bias_m_read_readvariableopdsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_136_kernel_m_read_readvariableopbsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_136_bias_m_read_readvariableopdsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_137_kernel_m_read_readvariableopbsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_137_bias_m_read_readvariableopdsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_138_kernel_m_read_readvariableopbsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_138_bias_m_read_readvariableop2savev2_adam_dense_139_kernel_m_read_readvariableop0savev2_adam_dense_139_bias_m_read_readvariableop2savev2_adam_dense_140_kernel_m_read_readvariableop0savev2_adam_dense_140_bias_m_read_readvariableopSsavev2_adam_transformer_block_15_layer_normalization_30_gamma_m_read_readvariableopRsavev2_adam_transformer_block_15_layer_normalization_30_beta_m_read_readvariableopSsavev2_adam_transformer_block_15_layer_normalization_31_gamma_m_read_readvariableopRsavev2_adam_transformer_block_15_layer_normalization_31_beta_m_read_readvariableop3savev2_adam_aux_output_kernel_v_read_readvariableop1savev2_adam_aux_output_bias_v_read_readvariableop2savev2_adam_dense_141_kernel_v_read_readvariableop0savev2_adam_dense_141_bias_v_read_readvariableop2savev2_adam_dense_142_kernel_v_read_readvariableop0savev2_adam_dense_142_bias_v_read_readvariableop2savev2_adam_dense_143_kernel_v_read_readvariableop0savev2_adam_dense_143_bias_v_read_readvariableop4savev2_adam_main_output_kernel_v_read_readvariableop2savev2_adam_main_output_bias_v_read_readvariableopYsavev2_adam_token_and_position_embedding_15_embedding_30_embeddings_v_read_readvariableopYsavev2_adam_token_and_position_embedding_15_embedding_31_embeddings_v_read_readvariableopdsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_135_kernel_v_read_readvariableopbsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_135_bias_v_read_readvariableopdsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_136_kernel_v_read_readvariableopbsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_136_bias_v_read_readvariableopdsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_137_kernel_v_read_readvariableopbsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_137_bias_v_read_readvariableopdsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_138_kernel_v_read_readvariableopbsavev2_adam_transformer_block_15_multi_head_self_attention_15_dense_138_bias_v_read_readvariableop2savev2_adam_dense_139_kernel_v_read_readvariableop0savev2_adam_dense_139_bias_v_read_readvariableop2savev2_adam_dense_140_kernel_v_read_readvariableop0savev2_adam_dense_140_bias_v_read_readvariableopSsavev2_adam_transformer_block_15_layer_normalization_30_gamma_v_read_readvariableopRsavev2_adam_transformer_block_15_layer_normalization_30_beta_v_read_readvariableopSsavev2_adam_transformer_block_15_layer_normalization_31_gamma_v_read_readvariableopRsavev2_adam_transformer_block_15_layer_normalization_31_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *r
dtypesh
f2d	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : ::@:@:@@:@:@@:@:@:: : : : : : :( :  : :  : :  : :  : :  : :  : : : : : : : : : : : : : : : : ::@:@:@@:@:@@:@:@:: :( :  : :  : :  : :  : :  : :  : : : : : : ::@:@:@@:@:@@:@:@:: :( :  : :  : :  : :  : :  : :  : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: :$ 

_output_shapes

:( :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :  

_output_shapes
: : !

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :$, 

_output_shapes

: : -

_output_shapes
::$. 

_output_shapes

:@: /

_output_shapes
:@:$0 

_output_shapes

:@@: 1

_output_shapes
:@:$2 

_output_shapes

:@@: 3

_output_shapes
:@:$4 

_output_shapes

:@: 5

_output_shapes
::$6 

_output_shapes

: :$7 

_output_shapes

:( :$8 

_output_shapes

:  : 9

_output_shapes
: :$: 

_output_shapes

:  : ;

_output_shapes
: :$< 

_output_shapes

:  : =

_output_shapes
: :$> 

_output_shapes

:  : ?

_output_shapes
: :$@ 

_output_shapes

:  : A

_output_shapes
: :$B 

_output_shapes

:  : C

_output_shapes
: : D

_output_shapes
: : E

_output_shapes
: : F

_output_shapes
: : G

_output_shapes
: :$H 

_output_shapes

: : I

_output_shapes
::$J 

_output_shapes

:@: K

_output_shapes
:@:$L 

_output_shapes

:@@: M

_output_shapes
:@:$N 

_output_shapes

:@@: O

_output_shapes
:@:$P 

_output_shapes

:@: Q

_output_shapes
::$R 

_output_shapes

: :$S 

_output_shapes

:( :$T 

_output_shapes

:  : U

_output_shapes
: :$V 

_output_shapes

:  : W

_output_shapes
: :$X 

_output_shapes

:  : Y

_output_shapes
: :$Z 

_output_shapes

:  : [

_output_shapes
: :$\ 

_output_shapes

:  : ]

_output_shapes
: :$^ 

_output_shapes

:  : _

_output_shapes
: : `

_output_shapes
: : a

_output_shapes
: : b

_output_shapes
: : c

_output_shapes
: :d

_output_shapes
: 
?
?
H__inference_main_output_layer_call_and_return_conditional_losses_6975595

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?%
"__inference__wrapped_model_6974963
input_16
	aux_input
aaindex_input`
Nmodel_15_token_and_position_embedding_15_embedding_31_embedding_lookup_6974670:( `
Nmodel_15_token_and_position_embedding_15_embedding_30_embedding_lookup_6974676: x
fmodel_15_transformer_block_15_multi_head_self_attention_15_dense_135_tensordot_readvariableop_resource:  r
dmodel_15_transformer_block_15_multi_head_self_attention_15_dense_135_biasadd_readvariableop_resource: x
fmodel_15_transformer_block_15_multi_head_self_attention_15_dense_136_tensordot_readvariableop_resource:  r
dmodel_15_transformer_block_15_multi_head_self_attention_15_dense_136_biasadd_readvariableop_resource: x
fmodel_15_transformer_block_15_multi_head_self_attention_15_dense_137_tensordot_readvariableop_resource:  r
dmodel_15_transformer_block_15_multi_head_self_attention_15_dense_137_biasadd_readvariableop_resource: x
fmodel_15_transformer_block_15_multi_head_self_attention_15_dense_138_tensordot_readvariableop_resource:  r
dmodel_15_transformer_block_15_multi_head_self_attention_15_dense_138_biasadd_readvariableop_resource: h
Zmodel_15_transformer_block_15_layer_normalization_30_batchnorm_mul_readvariableop_resource: d
Vmodel_15_transformer_block_15_layer_normalization_30_batchnorm_readvariableop_resource: i
Wmodel_15_transformer_block_15_sequential_15_dense_139_tensordot_readvariableop_resource:  c
Umodel_15_transformer_block_15_sequential_15_dense_139_biasadd_readvariableop_resource: i
Wmodel_15_transformer_block_15_sequential_15_dense_140_tensordot_readvariableop_resource:  c
Umodel_15_transformer_block_15_sequential_15_dense_140_biasadd_readvariableop_resource: h
Zmodel_15_transformer_block_15_layer_normalization_31_batchnorm_mul_readvariableop_resource: d
Vmodel_15_transformer_block_15_layer_normalization_31_batchnorm_readvariableop_resource: D
2model_15_aux_output_matmul_readvariableop_resource: A
3model_15_aux_output_biasadd_readvariableop_resource:C
1model_15_dense_141_matmul_readvariableop_resource:@@
2model_15_dense_141_biasadd_readvariableop_resource:@C
1model_15_dense_142_matmul_readvariableop_resource:@@@
2model_15_dense_142_biasadd_readvariableop_resource:@C
1model_15_dense_143_matmul_readvariableop_resource:@@@
2model_15_dense_143_biasadd_readvariableop_resource:@E
3model_15_main_output_matmul_readvariableop_resource:@B
4model_15_main_output_biasadd_readvariableop_resource:
identity

identity_1??*model_15/aux_output/BiasAdd/ReadVariableOp?)model_15/aux_output/MatMul/ReadVariableOp?)model_15/dense_141/BiasAdd/ReadVariableOp?(model_15/dense_141/MatMul/ReadVariableOp?)model_15/dense_142/BiasAdd/ReadVariableOp?(model_15/dense_142/MatMul/ReadVariableOp?)model_15/dense_143/BiasAdd/ReadVariableOp?(model_15/dense_143/MatMul/ReadVariableOp?+model_15/main_output/BiasAdd/ReadVariableOp?*model_15/main_output/MatMul/ReadVariableOp?Fmodel_15/token_and_position_embedding_15/embedding_30/embedding_lookup?Fmodel_15/token_and_position_embedding_15/embedding_31/embedding_lookup?Mmodel_15/transformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp?Qmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp?Mmodel_15/transformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp?Qmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp?[model_15/transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp?]model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?[model_15/transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp?]model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?[model_15/transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp?]model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?[model_15/transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp?]model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?Lmodel_15/transformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp?Nmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp?Lmodel_15/transformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp?Nmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp?
.model_15/token_and_position_embedding_15/ShapeShapeinput_16*
T0*
_output_shapes
:20
.model_15/token_and_position_embedding_15/Shape?
<model_15/token_and_position_embedding_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2>
<model_15/token_and_position_embedding_15/strided_slice/stack?
>model_15/token_and_position_embedding_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2@
>model_15/token_and_position_embedding_15/strided_slice/stack_1?
>model_15/token_and_position_embedding_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>model_15/token_and_position_embedding_15/strided_slice/stack_2?
6model_15/token_and_position_embedding_15/strided_sliceStridedSlice7model_15/token_and_position_embedding_15/Shape:output:0Emodel_15/token_and_position_embedding_15/strided_slice/stack:output:0Gmodel_15/token_and_position_embedding_15/strided_slice/stack_1:output:0Gmodel_15/token_and_position_embedding_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6model_15/token_and_position_embedding_15/strided_slice?
4model_15/token_and_position_embedding_15/range/startConst*
_output_shapes
: *
dtype0*
value	B : 26
4model_15/token_and_position_embedding_15/range/start?
4model_15/token_and_position_embedding_15/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :26
4model_15/token_and_position_embedding_15/range/delta?
.model_15/token_and_position_embedding_15/rangeRange=model_15/token_and_position_embedding_15/range/start:output:0?model_15/token_and_position_embedding_15/strided_slice:output:0=model_15/token_and_position_embedding_15/range/delta:output:0*#
_output_shapes
:?????????20
.model_15/token_and_position_embedding_15/range?
Fmodel_15/token_and_position_embedding_15/embedding_31/embedding_lookupResourceGatherNmodel_15_token_and_position_embedding_15_embedding_31_embedding_lookup_69746707model_15/token_and_position_embedding_15/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*a
_classW
USloc:@model_15/token_and_position_embedding_15/embedding_31/embedding_lookup/6974670*'
_output_shapes
:????????? *
dtype02H
Fmodel_15/token_and_position_embedding_15/embedding_31/embedding_lookup?
Omodel_15/token_and_position_embedding_15/embedding_31/embedding_lookup/IdentityIdentityOmodel_15/token_and_position_embedding_15/embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*a
_classW
USloc:@model_15/token_and_position_embedding_15/embedding_31/embedding_lookup/6974670*'
_output_shapes
:????????? 2Q
Omodel_15/token_and_position_embedding_15/embedding_31/embedding_lookup/Identity?
Qmodel_15/token_and_position_embedding_15/embedding_31/embedding_lookup/Identity_1IdentityXmodel_15/token_and_position_embedding_15/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2S
Qmodel_15/token_and_position_embedding_15/embedding_31/embedding_lookup/Identity_1?
:model_15/token_and_position_embedding_15/embedding_30/CastCastinput_16*

DstT0*

SrcT0*'
_output_shapes
:?????????(2<
:model_15/token_and_position_embedding_15/embedding_30/Cast?
Fmodel_15/token_and_position_embedding_15/embedding_30/embedding_lookupResourceGatherNmodel_15_token_and_position_embedding_15_embedding_30_embedding_lookup_6974676>model_15/token_and_position_embedding_15/embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*a
_classW
USloc:@model_15/token_and_position_embedding_15/embedding_30/embedding_lookup/6974676*+
_output_shapes
:?????????( *
dtype02H
Fmodel_15/token_and_position_embedding_15/embedding_30/embedding_lookup?
Omodel_15/token_and_position_embedding_15/embedding_30/embedding_lookup/IdentityIdentityOmodel_15/token_and_position_embedding_15/embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*a
_classW
USloc:@model_15/token_and_position_embedding_15/embedding_30/embedding_lookup/6974676*+
_output_shapes
:?????????( 2Q
Omodel_15/token_and_position_embedding_15/embedding_30/embedding_lookup/Identity?
Qmodel_15/token_and_position_embedding_15/embedding_30/embedding_lookup/Identity_1IdentityXmodel_15/token_and_position_embedding_15/embedding_30/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2S
Qmodel_15/token_and_position_embedding_15/embedding_30/embedding_lookup/Identity_1?
,model_15/token_and_position_embedding_15/addAddV2Zmodel_15/token_and_position_embedding_15/embedding_30/embedding_lookup/Identity_1:output:0Zmodel_15/token_and_position_embedding_15/embedding_31/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2.
,model_15/token_and_position_embedding_15/add?
@model_15/transformer_block_15/multi_head_self_attention_15/ShapeShape0model_15/token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:2B
@model_15/transformer_block_15/multi_head_self_attention_15/Shape?
Nmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2P
Nmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice/stack?
Pmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice/stack_1?
Pmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice/stack_2?
Hmodel_15/transformer_block_15/multi_head_self_attention_15/strided_sliceStridedSliceImodel_15/transformer_block_15/multi_head_self_attention_15/Shape:output:0Wmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice/stack:output:0Ymodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice/stack_1:output:0Ymodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice?
]model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpReadVariableOpfmodel_15_transformer_block_15_multi_head_self_attention_15_dense_135_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02_
]model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axes?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/free?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ShapeShape0model_15/token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Shape?
\model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axis?
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2GatherV2]model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/free:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2Y
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2?
^model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis?
Ymodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1GatherV2]model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axes:output:0gmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2[
Ymodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ProdProd`model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0]model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const:output:0*
T0*
_output_shapes
: 2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_1?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod_1Prodbmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1:output:0_model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod_1?
Zmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat/axis?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concatConcatV2\model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/free:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/axes:output:0cmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/stackPack\model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod:output:0^model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/stack?
Xmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/transpose	Transpose0model_15/token_and_position_embedding_15/add:z:0^model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2Z
Xmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/transpose?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReshapeReshape\model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/transpose:y:0]model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Reshape?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/MatMulMatMul_model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Reshape:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/MatMul?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_2?
\model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1/axis?
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1ConcatV2`model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0_model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/Const_2:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2Y
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1?
Nmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/TensordotReshape_model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/MatMul:product:0`model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2P
Nmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot?
[model_15/transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpReadVariableOpdmodel_15_transformer_block_15_multi_head_self_attention_15_dense_135_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02]
[model_15/transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/BiasAddBiasAddWmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot:output:0cmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd?
]model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpReadVariableOpfmodel_15_transformer_block_15_multi_head_self_attention_15_dense_136_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02_
]model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axes?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/free?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ShapeShape0model_15/token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Shape?
\model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axis?
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2GatherV2]model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/free:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2Y
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2?
^model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis?
Ymodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1GatherV2]model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axes:output:0gmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2[
Ymodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ProdProd`model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0]model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const:output:0*
T0*
_output_shapes
: 2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_1?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod_1Prodbmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1:output:0_model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod_1?
Zmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat/axis?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concatConcatV2\model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/free:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/axes:output:0cmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/stackPack\model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod:output:0^model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/stack?
Xmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/transpose	Transpose0model_15/token_and_position_embedding_15/add:z:0^model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2Z
Xmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/transpose?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReshapeReshape\model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/transpose:y:0]model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Reshape?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/MatMulMatMul_model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Reshape:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/MatMul?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_2?
\model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1/axis?
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1ConcatV2`model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0_model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/Const_2:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2Y
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1?
Nmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/TensordotReshape_model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/MatMul:product:0`model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2P
Nmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot?
[model_15/transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpReadVariableOpdmodel_15_transformer_block_15_multi_head_self_attention_15_dense_136_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02]
[model_15/transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/BiasAddBiasAddWmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot:output:0cmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd?
]model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpReadVariableOpfmodel_15_transformer_block_15_multi_head_self_attention_15_dense_137_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02_
]model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axes?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/free?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ShapeShape0model_15/token_and_position_embedding_15/add:z:0*
T0*
_output_shapes
:2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Shape?
\model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axis?
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2GatherV2]model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/free:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2Y
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2?
^model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis?
Ymodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1GatherV2]model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axes:output:0gmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2[
Ymodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ProdProd`model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0]model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const:output:0*
T0*
_output_shapes
: 2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_1?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod_1Prodbmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1:output:0_model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod_1?
Zmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat/axis?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concatConcatV2\model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/free:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/axes:output:0cmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/stackPack\model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod:output:0^model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/stack?
Xmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/transpose	Transpose0model_15/token_and_position_embedding_15/add:z:0^model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2Z
Xmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/transpose?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReshapeReshape\model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/transpose:y:0]model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Reshape?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/MatMulMatMul_model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Reshape:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/MatMul?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_2?
\model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1/axis?
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1ConcatV2`model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0_model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/Const_2:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2Y
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1?
Nmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/TensordotReshape_model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/MatMul:product:0`model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2P
Nmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot?
[model_15/transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpReadVariableOpdmodel_15_transformer_block_15_multi_head_self_attention_15_dense_137_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02]
[model_15/transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/BiasAddBiasAddWmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot:output:0cmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd?
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2L
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape/1?
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2L
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape/2?
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2L
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape/3?
Hmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shapePackQmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice:output:0Smodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape/1:output:0Smodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape/2:output:0Smodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2J
Hmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape?
Bmodel_15/transformer_block_15/multi_head_self_attention_15/ReshapeReshapeUmodel_15/transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd:output:0Qmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2D
Bmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape?
Imodel_15/transformer_block_15/multi_head_self_attention_15/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2K
Imodel_15/transformer_block_15/multi_head_self_attention_15/transpose/perm?
Dmodel_15/transformer_block_15/multi_head_self_attention_15/transpose	TransposeKmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape:output:0Rmodel_15/transformer_block_15/multi_head_self_attention_15/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2F
Dmodel_15/transformer_block_15/multi_head_self_attention_15/transpose?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape/1?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape/2?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape/3?
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shapePackQmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice:output:0Umodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape/1:output:0Umodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape/2:output:0Umodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2L
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape?
Dmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1ReshapeUmodel_15/transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd:output:0Smodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2F
Dmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1?
Kmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2M
Kmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_1/perm?
Fmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_1	TransposeMmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_1:output:0Tmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2H
Fmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_1?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape/1?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape/2?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape/3?
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shapePackQmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice:output:0Umodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape/1:output:0Umodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape/2:output:0Umodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2L
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape?
Dmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2ReshapeUmodel_15/transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd:output:0Smodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2F
Dmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2?
Kmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2M
Kmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_2/perm?
Fmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_2	TransposeMmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_2:output:0Tmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2H
Fmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_2?
Amodel_15/transformer_block_15/multi_head_self_attention_15/MatMulBatchMatMulV2Hmodel_15/transformer_block_15/multi_head_self_attention_15/transpose:y:0Jmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2C
Amodel_15/transformer_block_15/multi_head_self_attention_15/MatMul?
Bmodel_15/transformer_block_15/multi_head_self_attention_15/Shape_1ShapeJmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_1:y:0*
T0*
_output_shapes
:2D
Bmodel_15/transformer_block_15/multi_head_self_attention_15/Shape_1?
Pmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2R
Pmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1/stack?
Rmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_1?
Rmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2T
Rmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_2?
Jmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1StridedSliceKmodel_15/transformer_block_15/multi_head_self_attention_15/Shape_1:output:0Ymodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1/stack:output:0[model_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_1:output:0[model_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2L
Jmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1?
?model_15/transformer_block_15/multi_head_self_attention_15/CastCastSmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2A
?model_15/transformer_block_15/multi_head_self_attention_15/Cast?
?model_15/transformer_block_15/multi_head_self_attention_15/SqrtSqrtCmodel_15/transformer_block_15/multi_head_self_attention_15/Cast:y:0*
T0*
_output_shapes
: 2A
?model_15/transformer_block_15/multi_head_self_attention_15/Sqrt?
Bmodel_15/transformer_block_15/multi_head_self_attention_15/truedivRealDivJmodel_15/transformer_block_15/multi_head_self_attention_15/MatMul:output:0Cmodel_15/transformer_block_15/multi_head_self_attention_15/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2D
Bmodel_15/transformer_block_15/multi_head_self_attention_15/truediv?
Bmodel_15/transformer_block_15/multi_head_self_attention_15/SoftmaxSoftmaxFmodel_15/transformer_block_15/multi_head_self_attention_15/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2D
Bmodel_15/transformer_block_15/multi_head_self_attention_15/Softmax?
Cmodel_15/transformer_block_15/multi_head_self_attention_15/MatMul_1BatchMatMulV2Lmodel_15/transformer_block_15/multi_head_self_attention_15/Softmax:softmax:0Jmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2E
Cmodel_15/transformer_block_15/multi_head_self_attention_15/MatMul_1?
Kmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2M
Kmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_3/perm?
Fmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_3	TransposeLmodel_15/transformer_block_15/multi_head_self_attention_15/MatMul_1:output:0Tmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2H
Fmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_3?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3/shape/1?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3/shape/2?
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3/shapePackQmodel_15/transformer_block_15/multi_head_self_attention_15/strided_slice:output:0Umodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3/shape/1:output:0Umodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2L
Jmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3/shape?
Dmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3ReshapeJmodel_15/transformer_block_15/multi_head_self_attention_15/transpose_3:y:0Smodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2F
Dmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3?
]model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpReadVariableOpfmodel_15_transformer_block_15_multi_head_self_attention_15_dense_138_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02_
]model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axes?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/free?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ShapeShapeMmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3:output:0*
T0*
_output_shapes
:2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Shape?
\model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axis?
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2GatherV2]model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/free:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2Y
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2?
^model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2`
^model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis?
Ymodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1GatherV2]model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axes:output:0gmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2[
Ymodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const?
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ProdProd`model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0]model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const:output:0*
T0*
_output_shapes
: 2U
Smodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_1?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod_1Prodbmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1:output:0_model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod_1?
Zmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat/axis?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concatConcatV2\model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/free:output:0\model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/axes:output:0cmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat?
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/stackPack\model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod:output:0^model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2V
Tmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/stack?
Xmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/transpose	TransposeMmodel_15/transformer_block_15/multi_head_self_attention_15/Reshape_3:output:0^model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2Z
Xmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/transpose?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReshapeReshape\model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/transpose:y:0]model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Reshape?
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/MatMulMatMul_model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Reshape:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2W
Umodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/MatMul?
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2X
Vmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_2?
\model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2^
\model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1/axis?
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1ConcatV2`model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0_model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/Const_2:output:0emodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2Y
Wmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1?
Nmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/TensordotReshape_model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/MatMul:product:0`model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 2P
Nmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot?
[model_15/transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpReadVariableOpdmodel_15_transformer_block_15_multi_head_self_attention_15_dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02]
[model_15/transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp?
Lmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/BiasAddBiasAddWmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot:output:0cmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2N
Lmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd?
1model_15/transformer_block_15/dropout_30/IdentityIdentityUmodel_15/transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 23
1model_15/transformer_block_15/dropout_30/Identity?
!model_15/transformer_block_15/addAddV20model_15/token_and_position_embedding_15/add:z:0:model_15/transformer_block_15/dropout_30/Identity:output:0*
T0*+
_output_shapes
:?????????( 2#
!model_15/transformer_block_15/add?
Smodel_15/transformer_block_15/layer_normalization_30/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2U
Smodel_15/transformer_block_15/layer_normalization_30/moments/mean/reduction_indices?
Amodel_15/transformer_block_15/layer_normalization_30/moments/meanMean%model_15/transformer_block_15/add:z:0\model_15/transformer_block_15/layer_normalization_30/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2C
Amodel_15/transformer_block_15/layer_normalization_30/moments/mean?
Imodel_15/transformer_block_15/layer_normalization_30/moments/StopGradientStopGradientJmodel_15/transformer_block_15/layer_normalization_30/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2K
Imodel_15/transformer_block_15/layer_normalization_30/moments/StopGradient?
Nmodel_15/transformer_block_15/layer_normalization_30/moments/SquaredDifferenceSquaredDifference%model_15/transformer_block_15/add:z:0Rmodel_15/transformer_block_15/layer_normalization_30/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2P
Nmodel_15/transformer_block_15/layer_normalization_30/moments/SquaredDifference?
Wmodel_15/transformer_block_15/layer_normalization_30/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2Y
Wmodel_15/transformer_block_15/layer_normalization_30/moments/variance/reduction_indices?
Emodel_15/transformer_block_15/layer_normalization_30/moments/varianceMeanRmodel_15/transformer_block_15/layer_normalization_30/moments/SquaredDifference:z:0`model_15/transformer_block_15/layer_normalization_30/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2G
Emodel_15/transformer_block_15/layer_normalization_30/moments/variance?
Dmodel_15/transformer_block_15/layer_normalization_30/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52F
Dmodel_15/transformer_block_15/layer_normalization_30/batchnorm/add/y?
Bmodel_15/transformer_block_15/layer_normalization_30/batchnorm/addAddV2Nmodel_15/transformer_block_15/layer_normalization_30/moments/variance:output:0Mmodel_15/transformer_block_15/layer_normalization_30/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2D
Bmodel_15/transformer_block_15/layer_normalization_30/batchnorm/add?
Dmodel_15/transformer_block_15/layer_normalization_30/batchnorm/RsqrtRsqrtFmodel_15/transformer_block_15/layer_normalization_30/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2F
Dmodel_15/transformer_block_15/layer_normalization_30/batchnorm/Rsqrt?
Qmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOpReadVariableOpZmodel_15_transformer_block_15_layer_normalization_30_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02S
Qmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp?
Bmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mulMulHmodel_15/transformer_block_15/layer_normalization_30/batchnorm/Rsqrt:y:0Ymodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2D
Bmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul?
Dmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul_1Mul%model_15/transformer_block_15/add:z:0Fmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2F
Dmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul_1?
Dmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul_2MulJmodel_15/transformer_block_15/layer_normalization_30/moments/mean:output:0Fmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2F
Dmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul_2?
Mmodel_15/transformer_block_15/layer_normalization_30/batchnorm/ReadVariableOpReadVariableOpVmodel_15_transformer_block_15_layer_normalization_30_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02O
Mmodel_15/transformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp?
Bmodel_15/transformer_block_15/layer_normalization_30/batchnorm/subSubUmodel_15/transformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp:value:0Hmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2D
Bmodel_15/transformer_block_15/layer_normalization_30/batchnorm/sub?
Dmodel_15/transformer_block_15/layer_normalization_30/batchnorm/add_1AddV2Hmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul_1:z:0Fmodel_15/transformer_block_15/layer_normalization_30/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2F
Dmodel_15/transformer_block_15/layer_normalization_30/batchnorm/add_1?
Nmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOpReadVariableOpWmodel_15_transformer_block_15_sequential_15_dense_139_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02P
Nmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp?
Dmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2F
Dmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/axes?
Dmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/free?
Emodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/ShapeShapeHmodel_15/transformer_block_15/layer_normalization_30/batchnorm/add_1:z:0*
T0*
_output_shapes
:2G
Emodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Shape?
Mmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2/axis?
Hmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2GatherV2Nmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Shape:output:0Mmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/free:output:0Vmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Hmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2?
Omodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Q
Omodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1/axis?
Jmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1GatherV2Nmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Shape:output:0Mmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/axes:output:0Xmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2L
Jmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1?
Emodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2G
Emodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Const?
Dmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/ProdProdQmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2:output:0Nmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Const:output:0*
T0*
_output_shapes
: 2F
Dmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Prod?
Gmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Const_1?
Fmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Prod_1ProdSmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2_1:output:0Pmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2H
Fmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Prod_1?
Kmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat/axis?
Fmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concatConcatV2Mmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/free:output:0Mmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/axes:output:0Tmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2H
Fmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat?
Emodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/stackPackMmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Prod:output:0Omodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2G
Emodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/stack?
Imodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/transpose	TransposeHmodel_15/transformer_block_15/layer_normalization_30/batchnorm/add_1:z:0Omodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2K
Imodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/transpose?
Gmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/ReshapeReshapeMmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/transpose:y:0Nmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2I
Gmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Reshape?
Fmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/MatMulMatMulPmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Reshape:output:0Vmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2H
Fmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/MatMul?
Gmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Const_2?
Mmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat_1/axis?
Hmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat_1ConcatV2Qmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/GatherV2:output:0Pmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/Const_2:output:0Vmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2J
Hmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat_1?
?model_15/transformer_block_15/sequential_15/dense_139/TensordotReshapePmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/MatMul:product:0Qmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2A
?model_15/transformer_block_15/sequential_15/dense_139/Tensordot?
Lmodel_15/transformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOpReadVariableOpUmodel_15_transformer_block_15_sequential_15_dense_139_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02N
Lmodel_15/transformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp?
=model_15/transformer_block_15/sequential_15/dense_139/BiasAddBiasAddHmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot:output:0Tmodel_15/transformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2?
=model_15/transformer_block_15/sequential_15/dense_139/BiasAdd?
:model_15/transformer_block_15/sequential_15/dense_139/ReluReluFmodel_15/transformer_block_15/sequential_15/dense_139/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2<
:model_15/transformer_block_15/sequential_15/dense_139/Relu?
Nmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOpReadVariableOpWmodel_15_transformer_block_15_sequential_15_dense_140_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02P
Nmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp?
Dmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2F
Dmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/axes?
Dmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/free?
Emodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/ShapeShapeHmodel_15/transformer_block_15/sequential_15/dense_139/Relu:activations:0*
T0*
_output_shapes
:2G
Emodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Shape?
Mmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2/axis?
Hmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2GatherV2Nmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Shape:output:0Mmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/free:output:0Vmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Hmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2?
Omodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Q
Omodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1/axis?
Jmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1GatherV2Nmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Shape:output:0Mmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/axes:output:0Xmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2L
Jmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1?
Emodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2G
Emodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Const?
Dmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/ProdProdQmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2:output:0Nmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Const:output:0*
T0*
_output_shapes
: 2F
Dmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Prod?
Gmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Const_1?
Fmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Prod_1ProdSmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2_1:output:0Pmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2H
Fmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Prod_1?
Kmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat/axis?
Fmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concatConcatV2Mmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/free:output:0Mmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/axes:output:0Tmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2H
Fmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat?
Emodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/stackPackMmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Prod:output:0Omodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2G
Emodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/stack?
Imodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/transpose	TransposeHmodel_15/transformer_block_15/sequential_15/dense_139/Relu:activations:0Omodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2K
Imodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/transpose?
Gmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/ReshapeReshapeMmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/transpose:y:0Nmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2I
Gmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Reshape?
Fmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/MatMulMatMulPmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Reshape:output:0Vmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2H
Fmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/MatMul?
Gmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Const_2?
Mmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat_1/axis?
Hmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat_1ConcatV2Qmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/GatherV2:output:0Pmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/Const_2:output:0Vmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2J
Hmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat_1?
?model_15/transformer_block_15/sequential_15/dense_140/TensordotReshapePmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/MatMul:product:0Qmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2A
?model_15/transformer_block_15/sequential_15/dense_140/Tensordot?
Lmodel_15/transformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOpReadVariableOpUmodel_15_transformer_block_15_sequential_15_dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02N
Lmodel_15/transformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp?
=model_15/transformer_block_15/sequential_15/dense_140/BiasAddBiasAddHmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot:output:0Tmodel_15/transformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2?
=model_15/transformer_block_15/sequential_15/dense_140/BiasAdd?
1model_15/transformer_block_15/dropout_31/IdentityIdentityFmodel_15/transformer_block_15/sequential_15/dense_140/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 23
1model_15/transformer_block_15/dropout_31/Identity?
#model_15/transformer_block_15/add_1AddV2Hmodel_15/transformer_block_15/layer_normalization_30/batchnorm/add_1:z:0:model_15/transformer_block_15/dropout_31/Identity:output:0*
T0*+
_output_shapes
:?????????( 2%
#model_15/transformer_block_15/add_1?
Smodel_15/transformer_block_15/layer_normalization_31/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2U
Smodel_15/transformer_block_15/layer_normalization_31/moments/mean/reduction_indices?
Amodel_15/transformer_block_15/layer_normalization_31/moments/meanMean'model_15/transformer_block_15/add_1:z:0\model_15/transformer_block_15/layer_normalization_31/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2C
Amodel_15/transformer_block_15/layer_normalization_31/moments/mean?
Imodel_15/transformer_block_15/layer_normalization_31/moments/StopGradientStopGradientJmodel_15/transformer_block_15/layer_normalization_31/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2K
Imodel_15/transformer_block_15/layer_normalization_31/moments/StopGradient?
Nmodel_15/transformer_block_15/layer_normalization_31/moments/SquaredDifferenceSquaredDifference'model_15/transformer_block_15/add_1:z:0Rmodel_15/transformer_block_15/layer_normalization_31/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2P
Nmodel_15/transformer_block_15/layer_normalization_31/moments/SquaredDifference?
Wmodel_15/transformer_block_15/layer_normalization_31/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2Y
Wmodel_15/transformer_block_15/layer_normalization_31/moments/variance/reduction_indices?
Emodel_15/transformer_block_15/layer_normalization_31/moments/varianceMeanRmodel_15/transformer_block_15/layer_normalization_31/moments/SquaredDifference:z:0`model_15/transformer_block_15/layer_normalization_31/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2G
Emodel_15/transformer_block_15/layer_normalization_31/moments/variance?
Dmodel_15/transformer_block_15/layer_normalization_31/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52F
Dmodel_15/transformer_block_15/layer_normalization_31/batchnorm/add/y?
Bmodel_15/transformer_block_15/layer_normalization_31/batchnorm/addAddV2Nmodel_15/transformer_block_15/layer_normalization_31/moments/variance:output:0Mmodel_15/transformer_block_15/layer_normalization_31/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2D
Bmodel_15/transformer_block_15/layer_normalization_31/batchnorm/add?
Dmodel_15/transformer_block_15/layer_normalization_31/batchnorm/RsqrtRsqrtFmodel_15/transformer_block_15/layer_normalization_31/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2F
Dmodel_15/transformer_block_15/layer_normalization_31/batchnorm/Rsqrt?
Qmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOpReadVariableOpZmodel_15_transformer_block_15_layer_normalization_31_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02S
Qmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp?
Bmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mulMulHmodel_15/transformer_block_15/layer_normalization_31/batchnorm/Rsqrt:y:0Ymodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2D
Bmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul?
Dmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul_1Mul'model_15/transformer_block_15/add_1:z:0Fmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2F
Dmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul_1?
Dmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul_2MulJmodel_15/transformer_block_15/layer_normalization_31/moments/mean:output:0Fmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2F
Dmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul_2?
Mmodel_15/transformer_block_15/layer_normalization_31/batchnorm/ReadVariableOpReadVariableOpVmodel_15_transformer_block_15_layer_normalization_31_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02O
Mmodel_15/transformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp?
Bmodel_15/transformer_block_15/layer_normalization_31/batchnorm/subSubUmodel_15/transformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp:value:0Hmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2D
Bmodel_15/transformer_block_15/layer_normalization_31/batchnorm/sub?
Dmodel_15/transformer_block_15/layer_normalization_31/batchnorm/add_1AddV2Hmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul_1:z:0Fmodel_15/transformer_block_15/layer_normalization_31/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2F
Dmodel_15/transformer_block_15/layer_normalization_31/batchnorm/add_1?
;model_15/global_average_pooling1d_15/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2=
;model_15/global_average_pooling1d_15/Mean/reduction_indices?
)model_15/global_average_pooling1d_15/MeanMeanHmodel_15/transformer_block_15/layer_normalization_31/batchnorm/add_1:z:0Dmodel_15/global_average_pooling1d_15/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2+
)model_15/global_average_pooling1d_15/Mean?
)model_15/aux_output/MatMul/ReadVariableOpReadVariableOp2model_15_aux_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02+
)model_15/aux_output/MatMul/ReadVariableOp?
model_15/aux_output/MatMulMatMul2model_15/global_average_pooling1d_15/Mean:output:01model_15/aux_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_15/aux_output/MatMul?
*model_15/aux_output/BiasAdd/ReadVariableOpReadVariableOp3model_15_aux_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_15/aux_output/BiasAdd/ReadVariableOp?
model_15/aux_output/BiasAddBiasAdd$model_15/aux_output/MatMul:product:02model_15/aux_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_15/aux_output/BiasAdd?
model_15/aux_output/SigmoidSigmoid$model_15/aux_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_15/aux_output/Sigmoid?
#model_15/concatenate_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#model_15/concatenate_15/concat/axis?
model_15/concatenate_15/concatConcatV2model_15/aux_output/Sigmoid:y:0	aux_inputaaindex_input,model_15/concatenate_15/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2 
model_15/concatenate_15/concat?
(model_15/dense_141/MatMul/ReadVariableOpReadVariableOp1model_15_dense_141_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02*
(model_15/dense_141/MatMul/ReadVariableOp?
model_15/dense_141/MatMulMatMul'model_15/concatenate_15/concat:output:00model_15/dense_141/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_15/dense_141/MatMul?
)model_15/dense_141/BiasAdd/ReadVariableOpReadVariableOp2model_15_dense_141_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_15/dense_141/BiasAdd/ReadVariableOp?
model_15/dense_141/BiasAddBiasAdd#model_15/dense_141/MatMul:product:01model_15/dense_141/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_15/dense_141/BiasAdd?
model_15/dense_141/ReluRelu#model_15/dense_141/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_15/dense_141/Relu?
(model_15/dense_142/MatMul/ReadVariableOpReadVariableOp1model_15_dense_142_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(model_15/dense_142/MatMul/ReadVariableOp?
model_15/dense_142/MatMulMatMul%model_15/dense_141/Relu:activations:00model_15/dense_142/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_15/dense_142/MatMul?
)model_15/dense_142/BiasAdd/ReadVariableOpReadVariableOp2model_15_dense_142_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_15/dense_142/BiasAdd/ReadVariableOp?
model_15/dense_142/BiasAddBiasAdd#model_15/dense_142/MatMul:product:01model_15/dense_142/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_15/dense_142/BiasAdd?
model_15/dense_142/ReluRelu#model_15/dense_142/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_15/dense_142/Relu?
(model_15/dense_143/MatMul/ReadVariableOpReadVariableOp1model_15_dense_143_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(model_15/dense_143/MatMul/ReadVariableOp?
model_15/dense_143/MatMulMatMul%model_15/dense_142/Relu:activations:00model_15/dense_143/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_15/dense_143/MatMul?
)model_15/dense_143/BiasAdd/ReadVariableOpReadVariableOp2model_15_dense_143_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_15/dense_143/BiasAdd/ReadVariableOp?
model_15/dense_143/BiasAddBiasAdd#model_15/dense_143/MatMul:product:01model_15/dense_143/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_15/dense_143/BiasAdd?
model_15/dense_143/ReluRelu#model_15/dense_143/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_15/dense_143/Relu?
*model_15/main_output/MatMul/ReadVariableOpReadVariableOp3model_15_main_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*model_15/main_output/MatMul/ReadVariableOp?
model_15/main_output/MatMulMatMul%model_15/dense_143/Relu:activations:02model_15/main_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_15/main_output/MatMul?
+model_15/main_output/BiasAdd/ReadVariableOpReadVariableOp4model_15_main_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_15/main_output/BiasAdd/ReadVariableOp?
model_15/main_output/BiasAddBiasAdd%model_15/main_output/MatMul:product:03model_15/main_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_15/main_output/BiasAdd?
model_15/main_output/SigmoidSigmoid%model_15/main_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_15/main_output/Sigmoidz
IdentityIdentitymodel_15/aux_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity model_15/main_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp+^model_15/aux_output/BiasAdd/ReadVariableOp*^model_15/aux_output/MatMul/ReadVariableOp*^model_15/dense_141/BiasAdd/ReadVariableOp)^model_15/dense_141/MatMul/ReadVariableOp*^model_15/dense_142/BiasAdd/ReadVariableOp)^model_15/dense_142/MatMul/ReadVariableOp*^model_15/dense_143/BiasAdd/ReadVariableOp)^model_15/dense_143/MatMul/ReadVariableOp,^model_15/main_output/BiasAdd/ReadVariableOp+^model_15/main_output/MatMul/ReadVariableOpG^model_15/token_and_position_embedding_15/embedding_30/embedding_lookupG^model_15/token_and_position_embedding_15/embedding_31/embedding_lookupN^model_15/transformer_block_15/layer_normalization_30/batchnorm/ReadVariableOpR^model_15/transformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOpN^model_15/transformer_block_15/layer_normalization_31/batchnorm/ReadVariableOpR^model_15/transformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp\^model_15/transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp^^model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp\^model_15/transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp^^model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp\^model_15/transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp^^model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp\^model_15/transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp^^model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpM^model_15/transformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOpO^model_15/transformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOpM^model_15/transformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOpO^model_15/transformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*model_15/aux_output/BiasAdd/ReadVariableOp*model_15/aux_output/BiasAdd/ReadVariableOp2V
)model_15/aux_output/MatMul/ReadVariableOp)model_15/aux_output/MatMul/ReadVariableOp2V
)model_15/dense_141/BiasAdd/ReadVariableOp)model_15/dense_141/BiasAdd/ReadVariableOp2T
(model_15/dense_141/MatMul/ReadVariableOp(model_15/dense_141/MatMul/ReadVariableOp2V
)model_15/dense_142/BiasAdd/ReadVariableOp)model_15/dense_142/BiasAdd/ReadVariableOp2T
(model_15/dense_142/MatMul/ReadVariableOp(model_15/dense_142/MatMul/ReadVariableOp2V
)model_15/dense_143/BiasAdd/ReadVariableOp)model_15/dense_143/BiasAdd/ReadVariableOp2T
(model_15/dense_143/MatMul/ReadVariableOp(model_15/dense_143/MatMul/ReadVariableOp2Z
+model_15/main_output/BiasAdd/ReadVariableOp+model_15/main_output/BiasAdd/ReadVariableOp2X
*model_15/main_output/MatMul/ReadVariableOp*model_15/main_output/MatMul/ReadVariableOp2?
Fmodel_15/token_and_position_embedding_15/embedding_30/embedding_lookupFmodel_15/token_and_position_embedding_15/embedding_30/embedding_lookup2?
Fmodel_15/token_and_position_embedding_15/embedding_31/embedding_lookupFmodel_15/token_and_position_embedding_15/embedding_31/embedding_lookup2?
Mmodel_15/transformer_block_15/layer_normalization_30/batchnorm/ReadVariableOpMmodel_15/transformer_block_15/layer_normalization_30/batchnorm/ReadVariableOp2?
Qmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOpQmodel_15/transformer_block_15/layer_normalization_30/batchnorm/mul/ReadVariableOp2?
Mmodel_15/transformer_block_15/layer_normalization_31/batchnorm/ReadVariableOpMmodel_15/transformer_block_15/layer_normalization_31/batchnorm/ReadVariableOp2?
Qmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOpQmodel_15/transformer_block_15/layer_normalization_31/batchnorm/mul/ReadVariableOp2?
[model_15/transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp[model_15/transformer_block_15/multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp2?
]model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp]model_15/transformer_block_15/multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp2?
[model_15/transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp[model_15/transformer_block_15/multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp2?
]model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp]model_15/transformer_block_15/multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp2?
[model_15/transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp[model_15/transformer_block_15/multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp2?
]model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp]model_15/transformer_block_15/multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp2?
[model_15/transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp[model_15/transformer_block_15/multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp2?
]model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp]model_15/transformer_block_15/multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp2?
Lmodel_15/transformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOpLmodel_15/transformer_block_15/sequential_15/dense_139/BiasAdd/ReadVariableOp2?
Nmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOpNmodel_15/transformer_block_15/sequential_15/dense_139/Tensordot/ReadVariableOp2?
Lmodel_15/transformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOpLmodel_15/transformer_block_15/sequential_15/dense_140/BiasAdd/ReadVariableOp2?
Nmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOpNmodel_15/transformer_block_15/sequential_15/dense_140/Tensordot/ReadVariableOp:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
input_16:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input:VR
'
_output_shapes
:?????????
'
_user_specified_nameaaindex_input
?
?
F__inference_dense_143_layer_call_and_return_conditional_losses_6975578

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
F__inference_dense_142_layer_call_and_return_conditional_losses_6978007

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_model_15_layer_call_fn_6976337
input_16
	aux_input
aaindex_input
unknown:( 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18:

unknown_19:@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_16	aux_inputaaindex_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_15_layer_call_and_return_conditional_losses_69762112
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????(:?????????:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
input_16:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input:VR
'
_output_shapes
:?????????
'
_user_specified_nameaaindex_input
?
t
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_6975166

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?L
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_6978113

inputs=
+dense_139_tensordot_readvariableop_resource:  7
)dense_139_biasadd_readvariableop_resource: =
+dense_140_tensordot_readvariableop_resource:  7
)dense_140_biasadd_readvariableop_resource: 
identity?? dense_139/BiasAdd/ReadVariableOp?"dense_139/Tensordot/ReadVariableOp? dense_140/BiasAdd/ReadVariableOp?"dense_140/Tensordot/ReadVariableOp?
"dense_139/Tensordot/ReadVariableOpReadVariableOp+dense_139_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02$
"dense_139/Tensordot/ReadVariableOp~
dense_139/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_139/Tensordot/axes?
dense_139/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_139/Tensordot/freel
dense_139/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_139/Tensordot/Shape?
!dense_139/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_139/Tensordot/GatherV2/axis?
dense_139/Tensordot/GatherV2GatherV2"dense_139/Tensordot/Shape:output:0!dense_139/Tensordot/free:output:0*dense_139/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_139/Tensordot/GatherV2?
#dense_139/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_139/Tensordot/GatherV2_1/axis?
dense_139/Tensordot/GatherV2_1GatherV2"dense_139/Tensordot/Shape:output:0!dense_139/Tensordot/axes:output:0,dense_139/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_139/Tensordot/GatherV2_1?
dense_139/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_139/Tensordot/Const?
dense_139/Tensordot/ProdProd%dense_139/Tensordot/GatherV2:output:0"dense_139/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_139/Tensordot/Prod?
dense_139/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_139/Tensordot/Const_1?
dense_139/Tensordot/Prod_1Prod'dense_139/Tensordot/GatherV2_1:output:0$dense_139/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_139/Tensordot/Prod_1?
dense_139/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_139/Tensordot/concat/axis?
dense_139/Tensordot/concatConcatV2!dense_139/Tensordot/free:output:0!dense_139/Tensordot/axes:output:0(dense_139/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_139/Tensordot/concat?
dense_139/Tensordot/stackPack!dense_139/Tensordot/Prod:output:0#dense_139/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_139/Tensordot/stack?
dense_139/Tensordot/transpose	Transposeinputs#dense_139/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_139/Tensordot/transpose?
dense_139/Tensordot/ReshapeReshape!dense_139/Tensordot/transpose:y:0"dense_139/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_139/Tensordot/Reshape?
dense_139/Tensordot/MatMulMatMul$dense_139/Tensordot/Reshape:output:0*dense_139/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_139/Tensordot/MatMul?
dense_139/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_139/Tensordot/Const_2?
!dense_139/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_139/Tensordot/concat_1/axis?
dense_139/Tensordot/concat_1ConcatV2%dense_139/Tensordot/GatherV2:output:0$dense_139/Tensordot/Const_2:output:0*dense_139/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_139/Tensordot/concat_1?
dense_139/TensordotReshape$dense_139/Tensordot/MatMul:product:0%dense_139/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_139/Tensordot?
 dense_139/BiasAdd/ReadVariableOpReadVariableOp)dense_139_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_139/BiasAdd/ReadVariableOp?
dense_139/BiasAddBiasAdddense_139/Tensordot:output:0(dense_139/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_139/BiasAddz
dense_139/ReluReludense_139/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dense_139/Relu?
"dense_140/Tensordot/ReadVariableOpReadVariableOp+dense_140_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02$
"dense_140/Tensordot/ReadVariableOp~
dense_140/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_140/Tensordot/axes?
dense_140/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_140/Tensordot/free?
dense_140/Tensordot/ShapeShapedense_139/Relu:activations:0*
T0*
_output_shapes
:2
dense_140/Tensordot/Shape?
!dense_140/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_140/Tensordot/GatherV2/axis?
dense_140/Tensordot/GatherV2GatherV2"dense_140/Tensordot/Shape:output:0!dense_140/Tensordot/free:output:0*dense_140/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_140/Tensordot/GatherV2?
#dense_140/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_140/Tensordot/GatherV2_1/axis?
dense_140/Tensordot/GatherV2_1GatherV2"dense_140/Tensordot/Shape:output:0!dense_140/Tensordot/axes:output:0,dense_140/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_140/Tensordot/GatherV2_1?
dense_140/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_140/Tensordot/Const?
dense_140/Tensordot/ProdProd%dense_140/Tensordot/GatherV2:output:0"dense_140/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_140/Tensordot/Prod?
dense_140/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_140/Tensordot/Const_1?
dense_140/Tensordot/Prod_1Prod'dense_140/Tensordot/GatherV2_1:output:0$dense_140/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_140/Tensordot/Prod_1?
dense_140/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_140/Tensordot/concat/axis?
dense_140/Tensordot/concatConcatV2!dense_140/Tensordot/free:output:0!dense_140/Tensordot/axes:output:0(dense_140/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_140/Tensordot/concat?
dense_140/Tensordot/stackPack!dense_140/Tensordot/Prod:output:0#dense_140/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_140/Tensordot/stack?
dense_140/Tensordot/transpose	Transposedense_139/Relu:activations:0#dense_140/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_140/Tensordot/transpose?
dense_140/Tensordot/ReshapeReshape!dense_140/Tensordot/transpose:y:0"dense_140/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_140/Tensordot/Reshape?
dense_140/Tensordot/MatMulMatMul$dense_140/Tensordot/Reshape:output:0*dense_140/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_140/Tensordot/MatMul?
dense_140/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_140/Tensordot/Const_2?
!dense_140/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_140/Tensordot/concat_1/axis?
dense_140/Tensordot/concat_1ConcatV2%dense_140/Tensordot/GatherV2:output:0$dense_140/Tensordot/Const_2:output:0*dense_140/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_140/Tensordot/concat_1?
dense_140/TensordotReshape$dense_140/Tensordot/MatMul:product:0%dense_140/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_140/Tensordot?
 dense_140/BiasAdd/ReadVariableOpReadVariableOp)dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_140/BiasAdd/ReadVariableOp?
dense_140/BiasAddBiasAdddense_140/Tensordot:output:0(dense_140/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_140/BiasAddy
IdentityIdentitydense_140/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp!^dense_139/BiasAdd/ReadVariableOp#^dense_139/Tensordot/ReadVariableOp!^dense_140/BiasAdd/ReadVariableOp#^dense_140/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2D
 dense_139/BiasAdd/ReadVariableOp dense_139/BiasAdd/ReadVariableOp2H
"dense_139/Tensordot/ReadVariableOp"dense_139/Tensordot/ReadVariableOp2D
 dense_140/BiasAdd/ReadVariableOp dense_140/BiasAdd/ReadVariableOp2H
"dense_140/Tensordot/ReadVariableOp"dense_140/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
t
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_6975504

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????( :S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
??
?
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_6977587

inputsZ
Hmulti_head_self_attention_15_dense_135_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_135_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_136_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_136_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_137_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_137_biasadd_readvariableop_resource: Z
Hmulti_head_self_attention_15_dense_138_tensordot_readvariableop_resource:  T
Fmulti_head_self_attention_15_dense_138_biasadd_readvariableop_resource: J
<layer_normalization_30_batchnorm_mul_readvariableop_resource: F
8layer_normalization_30_batchnorm_readvariableop_resource: K
9sequential_15_dense_139_tensordot_readvariableop_resource:  E
7sequential_15_dense_139_biasadd_readvariableop_resource: K
9sequential_15_dense_140_tensordot_readvariableop_resource:  E
7sequential_15_dense_140_biasadd_readvariableop_resource: J
<layer_normalization_31_batchnorm_mul_readvariableop_resource: F
8layer_normalization_31_batchnorm_readvariableop_resource: 
identity??/layer_normalization_30/batchnorm/ReadVariableOp?3layer_normalization_30/batchnorm/mul/ReadVariableOp?/layer_normalization_31/batchnorm/ReadVariableOp?3layer_normalization_31/batchnorm/mul/ReadVariableOp?=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp??multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?.sequential_15/dense_139/BiasAdd/ReadVariableOp?0sequential_15/dense_139/Tensordot/ReadVariableOp?.sequential_15/dense_140/BiasAdd/ReadVariableOp?0sequential_15/dense_140/Tensordot/ReadVariableOp~
"multi_head_self_attention_15/ShapeShapeinputs*
T0*
_output_shapes
:2$
"multi_head_self_attention_15/Shape?
0multi_head_self_attention_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0multi_head_self_attention_15/strided_slice/stack?
2multi_head_self_attention_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2multi_head_self_attention_15/strided_slice/stack_1?
2multi_head_self_attention_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2multi_head_self_attention_15/strided_slice/stack_2?
*multi_head_self_attention_15/strided_sliceStridedSlice+multi_head_self_attention_15/Shape:output:09multi_head_self_attention_15/strided_slice/stack:output:0;multi_head_self_attention_15/strided_slice/stack_1:output:0;multi_head_self_attention_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*multi_head_self_attention_15/strided_slice?
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_135_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_135/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_135/Tensordot/axes?
5multi_head_self_attention_15/dense_135/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_135/Tensordot/free?
6multi_head_self_attention_15/dense_135/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_135/Tensordot/Shape?
>multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_135/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_135/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_135/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_135/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_135/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_135/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_135/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_135/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_135/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_135/Tensordot/Const?
5multi_head_self_attention_15/dense_135/Tensordot/ProdProdBmulti_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_135/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_135/Tensordot/Prod?
8multi_head_self_attention_15/dense_135/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_135/Tensordot/Const_1?
7multi_head_self_attention_15/dense_135/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_135/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_135/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_135/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_135/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_135/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_135/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_135/Tensordot/free:output:0>multi_head_self_attention_15/dense_135/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_135/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_135/Tensordot/concat?
6multi_head_self_attention_15/dense_135/Tensordot/stackPack>multi_head_self_attention_15/dense_135/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_135/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_135/Tensordot/stack?
:multi_head_self_attention_15/dense_135/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_135/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_135/Tensordot/transpose?
8multi_head_self_attention_15/dense_135/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_135/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_135/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_135/Tensordot/Reshape?
7multi_head_self_attention_15/dense_135/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_135/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_135/Tensordot/MatMul?
8multi_head_self_attention_15/dense_135/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_135/Tensordot/Const_2?
>multi_head_self_attention_15/dense_135/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_135/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_135/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_135/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_135/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_135/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_135/Tensordot/concat_1?
0multi_head_self_attention_15/dense_135/TensordotReshapeAmulti_head_self_attention_15/dense_135/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_135/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_135/Tensordot?
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_135_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_135/BiasAddBiasAdd9multi_head_self_attention_15/dense_135/Tensordot:output:0Emulti_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_135/BiasAdd?
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_136_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_136/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_136/Tensordot/axes?
5multi_head_self_attention_15/dense_136/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_136/Tensordot/free?
6multi_head_self_attention_15/dense_136/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_136/Tensordot/Shape?
>multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_136/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_136/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_136/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_136/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_136/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_136/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_136/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_136/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_136/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_136/Tensordot/Const?
5multi_head_self_attention_15/dense_136/Tensordot/ProdProdBmulti_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_136/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_136/Tensordot/Prod?
8multi_head_self_attention_15/dense_136/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_136/Tensordot/Const_1?
7multi_head_self_attention_15/dense_136/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_136/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_136/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_136/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_136/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_136/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_136/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_136/Tensordot/free:output:0>multi_head_self_attention_15/dense_136/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_136/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_136/Tensordot/concat?
6multi_head_self_attention_15/dense_136/Tensordot/stackPack>multi_head_self_attention_15/dense_136/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_136/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_136/Tensordot/stack?
:multi_head_self_attention_15/dense_136/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_136/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_136/Tensordot/transpose?
8multi_head_self_attention_15/dense_136/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_136/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_136/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_136/Tensordot/Reshape?
7multi_head_self_attention_15/dense_136/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_136/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_136/Tensordot/MatMul?
8multi_head_self_attention_15/dense_136/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_136/Tensordot/Const_2?
>multi_head_self_attention_15/dense_136/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_136/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_136/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_136/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_136/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_136/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_136/Tensordot/concat_1?
0multi_head_self_attention_15/dense_136/TensordotReshapeAmulti_head_self_attention_15/dense_136/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_136/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_136/Tensordot?
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_136_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_136/BiasAddBiasAdd9multi_head_self_attention_15/dense_136/Tensordot:output:0Emulti_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_136/BiasAdd?
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_137_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_137/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_137/Tensordot/axes?
5multi_head_self_attention_15/dense_137/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_137/Tensordot/free?
6multi_head_self_attention_15/dense_137/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_137/Tensordot/Shape?
>multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_137/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_137/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_137/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_137/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_137/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_137/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_137/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_137/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_137/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_137/Tensordot/Const?
5multi_head_self_attention_15/dense_137/Tensordot/ProdProdBmulti_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_137/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_137/Tensordot/Prod?
8multi_head_self_attention_15/dense_137/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_137/Tensordot/Const_1?
7multi_head_self_attention_15/dense_137/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_137/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_137/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_137/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_137/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_137/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_137/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_137/Tensordot/free:output:0>multi_head_self_attention_15/dense_137/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_137/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_137/Tensordot/concat?
6multi_head_self_attention_15/dense_137/Tensordot/stackPack>multi_head_self_attention_15/dense_137/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_137/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_137/Tensordot/stack?
:multi_head_self_attention_15/dense_137/Tensordot/transpose	Transposeinputs@multi_head_self_attention_15/dense_137/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2<
:multi_head_self_attention_15/dense_137/Tensordot/transpose?
8multi_head_self_attention_15/dense_137/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_137/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_137/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_137/Tensordot/Reshape?
7multi_head_self_attention_15/dense_137/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_137/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_137/Tensordot/MatMul?
8multi_head_self_attention_15/dense_137/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_137/Tensordot/Const_2?
>multi_head_self_attention_15/dense_137/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_137/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_137/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_137/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_137/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_137/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_137/Tensordot/concat_1?
0multi_head_self_attention_15/dense_137/TensordotReshapeAmulti_head_self_attention_15/dense_137/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_137/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 22
0multi_head_self_attention_15/dense_137/Tensordot?
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_137_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_137/BiasAddBiasAdd9multi_head_self_attention_15/dense_137/Tensordot:output:0Emulti_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_15/dense_137/BiasAdd?
,multi_head_self_attention_15/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2.
,multi_head_self_attention_15/Reshape/shape/1?
,multi_head_self_attention_15/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2.
,multi_head_self_attention_15/Reshape/shape/2?
,multi_head_self_attention_15/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2.
,multi_head_self_attention_15/Reshape/shape/3?
*multi_head_self_attention_15/Reshape/shapePack3multi_head_self_attention_15/strided_slice:output:05multi_head_self_attention_15/Reshape/shape/1:output:05multi_head_self_attention_15/Reshape/shape/2:output:05multi_head_self_attention_15/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2,
*multi_head_self_attention_15/Reshape/shape?
$multi_head_self_attention_15/ReshapeReshape7multi_head_self_attention_15/dense_135/BiasAdd:output:03multi_head_self_attention_15/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_15/Reshape?
+multi_head_self_attention_15/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2-
+multi_head_self_attention_15/transpose/perm?
&multi_head_self_attention_15/transpose	Transpose-multi_head_self_attention_15/Reshape:output:04multi_head_self_attention_15/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/transpose?
.multi_head_self_attention_15/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_1/shape/1?
.multi_head_self_attention_15/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_1/shape/2?
.multi_head_self_attention_15/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_1/shape/3?
,multi_head_self_attention_15/Reshape_1/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_1/shape/1:output:07multi_head_self_attention_15/Reshape_1/shape/2:output:07multi_head_self_attention_15/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_1/shape?
&multi_head_self_attention_15/Reshape_1Reshape7multi_head_self_attention_15/dense_136/BiasAdd:output:05multi_head_self_attention_15/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/Reshape_1?
-multi_head_self_attention_15/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_1/perm?
(multi_head_self_attention_15/transpose_1	Transpose/multi_head_self_attention_15/Reshape_1:output:06multi_head_self_attention_15/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_1?
.multi_head_self_attention_15/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_2/shape/1?
.multi_head_self_attention_15/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_2/shape/2?
.multi_head_self_attention_15/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.multi_head_self_attention_15/Reshape_2/shape/3?
,multi_head_self_attention_15/Reshape_2/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_2/shape/1:output:07multi_head_self_attention_15/Reshape_2/shape/2:output:07multi_head_self_attention_15/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_2/shape?
&multi_head_self_attention_15/Reshape_2Reshape7multi_head_self_attention_15/dense_137/BiasAdd:output:05multi_head_self_attention_15/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2(
&multi_head_self_attention_15/Reshape_2?
-multi_head_self_attention_15/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_2/perm?
(multi_head_self_attention_15/transpose_2	Transpose/multi_head_self_attention_15/Reshape_2:output:06multi_head_self_attention_15/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_2?
#multi_head_self_attention_15/MatMulBatchMatMulV2*multi_head_self_attention_15/transpose:y:0,multi_head_self_attention_15/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2%
#multi_head_self_attention_15/MatMul?
$multi_head_self_attention_15/Shape_1Shape,multi_head_self_attention_15/transpose_1:y:0*
T0*
_output_shapes
:2&
$multi_head_self_attention_15/Shape_1?
2multi_head_self_attention_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2multi_head_self_attention_15/strided_slice_1/stack?
4multi_head_self_attention_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_15/strided_slice_1/stack_1?
4multi_head_self_attention_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4multi_head_self_attention_15/strided_slice_1/stack_2?
,multi_head_self_attention_15/strided_slice_1StridedSlice-multi_head_self_attention_15/Shape_1:output:0;multi_head_self_attention_15/strided_slice_1/stack:output:0=multi_head_self_attention_15/strided_slice_1/stack_1:output:0=multi_head_self_attention_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,multi_head_self_attention_15/strided_slice_1?
!multi_head_self_attention_15/CastCast5multi_head_self_attention_15/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!multi_head_self_attention_15/Cast?
!multi_head_self_attention_15/SqrtSqrt%multi_head_self_attention_15/Cast:y:0*
T0*
_output_shapes
: 2#
!multi_head_self_attention_15/Sqrt?
$multi_head_self_attention_15/truedivRealDiv,multi_head_self_attention_15/MatMul:output:0%multi_head_self_attention_15/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2&
$multi_head_self_attention_15/truediv?
$multi_head_self_attention_15/SoftmaxSoftmax(multi_head_self_attention_15/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2&
$multi_head_self_attention_15/Softmax?
%multi_head_self_attention_15/MatMul_1BatchMatMulV2.multi_head_self_attention_15/Softmax:softmax:0,multi_head_self_attention_15/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_15/MatMul_1?
-multi_head_self_attention_15/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-multi_head_self_attention_15/transpose_3/perm?
(multi_head_self_attention_15/transpose_3	Transpose.multi_head_self_attention_15/MatMul_1:output:06multi_head_self_attention_15/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2*
(multi_head_self_attention_15/transpose_3?
.multi_head_self_attention_15/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????20
.multi_head_self_attention_15/Reshape_3/shape/1?
.multi_head_self_attention_15/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 20
.multi_head_self_attention_15/Reshape_3/shape/2?
,multi_head_self_attention_15/Reshape_3/shapePack3multi_head_self_attention_15/strided_slice:output:07multi_head_self_attention_15/Reshape_3/shape/1:output:07multi_head_self_attention_15/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2.
,multi_head_self_attention_15/Reshape_3/shape?
&multi_head_self_attention_15/Reshape_3Reshape,multi_head_self_attention_15/transpose_3:y:05multi_head_self_attention_15/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2(
&multi_head_self_attention_15/Reshape_3?
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOpReadVariableOpHmulti_head_self_attention_15_dense_138_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02A
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?
5multi_head_self_attention_15/dense_138/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:27
5multi_head_self_attention_15/dense_138/Tensordot/axes?
5multi_head_self_attention_15/dense_138/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       27
5multi_head_self_attention_15/dense_138/Tensordot/free?
6multi_head_self_attention_15/dense_138/Tensordot/ShapeShape/multi_head_self_attention_15/Reshape_3:output:0*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_138/Tensordot/Shape?
>multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_138/Tensordot/GatherV2/axis?
9multi_head_self_attention_15/dense_138/Tensordot/GatherV2GatherV2?multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_138/Tensordot/free:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_138/Tensordot/GatherV2?
@multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis?
;multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1GatherV2?multi_head_self_attention_15/dense_138/Tensordot/Shape:output:0>multi_head_self_attention_15/dense_138/Tensordot/axes:output:0Imulti_head_self_attention_15/dense_138/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2=
;multi_head_self_attention_15/dense_138/Tensordot/GatherV2_1?
6multi_head_self_attention_15/dense_138/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_15/dense_138/Tensordot/Const?
5multi_head_self_attention_15/dense_138/Tensordot/ProdProdBmulti_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0?multi_head_self_attention_15/dense_138/Tensordot/Const:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_15/dense_138/Tensordot/Prod?
8multi_head_self_attention_15/dense_138/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_138/Tensordot/Const_1?
7multi_head_self_attention_15/dense_138/Tensordot/Prod_1ProdDmulti_head_self_attention_15/dense_138/Tensordot/GatherV2_1:output:0Amulti_head_self_attention_15/dense_138/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 29
7multi_head_self_attention_15/dense_138/Tensordot/Prod_1?
<multi_head_self_attention_15/dense_138/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_15/dense_138/Tensordot/concat/axis?
7multi_head_self_attention_15/dense_138/Tensordot/concatConcatV2>multi_head_self_attention_15/dense_138/Tensordot/free:output:0>multi_head_self_attention_15/dense_138/Tensordot/axes:output:0Emulti_head_self_attention_15/dense_138/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_15/dense_138/Tensordot/concat?
6multi_head_self_attention_15/dense_138/Tensordot/stackPack>multi_head_self_attention_15/dense_138/Tensordot/Prod:output:0@multi_head_self_attention_15/dense_138/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:28
6multi_head_self_attention_15/dense_138/Tensordot/stack?
:multi_head_self_attention_15/dense_138/Tensordot/transpose	Transpose/multi_head_self_attention_15/Reshape_3:output:0@multi_head_self_attention_15/dense_138/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2<
:multi_head_self_attention_15/dense_138/Tensordot/transpose?
8multi_head_self_attention_15/dense_138/Tensordot/ReshapeReshape>multi_head_self_attention_15/dense_138/Tensordot/transpose:y:0?multi_head_self_attention_15/dense_138/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2:
8multi_head_self_attention_15/dense_138/Tensordot/Reshape?
7multi_head_self_attention_15/dense_138/Tensordot/MatMulMatMulAmulti_head_self_attention_15/dense_138/Tensordot/Reshape:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 29
7multi_head_self_attention_15/dense_138/Tensordot/MatMul?
8multi_head_self_attention_15/dense_138/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2:
8multi_head_self_attention_15/dense_138/Tensordot/Const_2?
>multi_head_self_attention_15/dense_138/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_15/dense_138/Tensordot/concat_1/axis?
9multi_head_self_attention_15/dense_138/Tensordot/concat_1ConcatV2Bmulti_head_self_attention_15/dense_138/Tensordot/GatherV2:output:0Amulti_head_self_attention_15/dense_138/Tensordot/Const_2:output:0Gmulti_head_self_attention_15/dense_138/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2;
9multi_head_self_attention_15/dense_138/Tensordot/concat_1?
0multi_head_self_attention_15/dense_138/TensordotReshapeAmulti_head_self_attention_15/dense_138/Tensordot/MatMul:product:0Bmulti_head_self_attention_15/dense_138/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 22
0multi_head_self_attention_15/dense_138/Tensordot?
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOpReadVariableOpFmulti_head_self_attention_15_dense_138_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp?
.multi_head_self_attention_15/dense_138/BiasAddBiasAdd9multi_head_self_attention_15/dense_138/Tensordot:output:0Emulti_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_15/dense_138/BiasAdd?
dropout_30/IdentityIdentity7multi_head_self_attention_15/dense_138/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_30/Identityo
addAddV2inputsdropout_30/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add?
5layer_normalization_30/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_30/moments/mean/reduction_indices?
#layer_normalization_30/moments/meanMeanadd:z:0>layer_normalization_30/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_30/moments/mean?
+layer_normalization_30/moments/StopGradientStopGradient,layer_normalization_30/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_30/moments/StopGradient?
0layer_normalization_30/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_30/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_30/moments/SquaredDifference?
9layer_normalization_30/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_30/moments/variance/reduction_indices?
'layer_normalization_30/moments/varianceMean4layer_normalization_30/moments/SquaredDifference:z:0Blayer_normalization_30/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_30/moments/variance?
&layer_normalization_30/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_30/batchnorm/add/y?
$layer_normalization_30/batchnorm/addAddV20layer_normalization_30/moments/variance:output:0/layer_normalization_30/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_30/batchnorm/add?
&layer_normalization_30/batchnorm/RsqrtRsqrt(layer_normalization_30/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_30/batchnorm/Rsqrt?
3layer_normalization_30/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_30_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_30/batchnorm/mul/ReadVariableOp?
$layer_normalization_30/batchnorm/mulMul*layer_normalization_30/batchnorm/Rsqrt:y:0;layer_normalization_30/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_30/batchnorm/mul?
&layer_normalization_30/batchnorm/mul_1Muladd:z:0(layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/mul_1?
&layer_normalization_30/batchnorm/mul_2Mul,layer_normalization_30/moments/mean:output:0(layer_normalization_30/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/mul_2?
/layer_normalization_30/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_30_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_30/batchnorm/ReadVariableOp?
$layer_normalization_30/batchnorm/subSub7layer_normalization_30/batchnorm/ReadVariableOp:value:0*layer_normalization_30/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_30/batchnorm/sub?
&layer_normalization_30/batchnorm/add_1AddV2*layer_normalization_30/batchnorm/mul_1:z:0(layer_normalization_30/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_30/batchnorm/add_1?
0sequential_15/dense_139/Tensordot/ReadVariableOpReadVariableOp9sequential_15_dense_139_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype022
0sequential_15/dense_139/Tensordot/ReadVariableOp?
&sequential_15/dense_139/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_15/dense_139/Tensordot/axes?
&sequential_15/dense_139/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&sequential_15/dense_139/Tensordot/free?
'sequential_15/dense_139/Tensordot/ShapeShape*layer_normalization_30/batchnorm/add_1:z:0*
T0*
_output_shapes
:2)
'sequential_15/dense_139/Tensordot/Shape?
/sequential_15/dense_139/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_139/Tensordot/GatherV2/axis?
*sequential_15/dense_139/Tensordot/GatherV2GatherV20sequential_15/dense_139/Tensordot/Shape:output:0/sequential_15/dense_139/Tensordot/free:output:08sequential_15/dense_139/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_15/dense_139/Tensordot/GatherV2?
1sequential_15/dense_139/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_15/dense_139/Tensordot/GatherV2_1/axis?
,sequential_15/dense_139/Tensordot/GatherV2_1GatherV20sequential_15/dense_139/Tensordot/Shape:output:0/sequential_15/dense_139/Tensordot/axes:output:0:sequential_15/dense_139/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,sequential_15/dense_139/Tensordot/GatherV2_1?
'sequential_15/dense_139/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_15/dense_139/Tensordot/Const?
&sequential_15/dense_139/Tensordot/ProdProd3sequential_15/dense_139/Tensordot/GatherV2:output:00sequential_15/dense_139/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&sequential_15/dense_139/Tensordot/Prod?
)sequential_15/dense_139/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_139/Tensordot/Const_1?
(sequential_15/dense_139/Tensordot/Prod_1Prod5sequential_15/dense_139/Tensordot/GatherV2_1:output:02sequential_15/dense_139/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(sequential_15/dense_139/Tensordot/Prod_1?
-sequential_15/dense_139/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_15/dense_139/Tensordot/concat/axis?
(sequential_15/dense_139/Tensordot/concatConcatV2/sequential_15/dense_139/Tensordot/free:output:0/sequential_15/dense_139/Tensordot/axes:output:06sequential_15/dense_139/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_15/dense_139/Tensordot/concat?
'sequential_15/dense_139/Tensordot/stackPack/sequential_15/dense_139/Tensordot/Prod:output:01sequential_15/dense_139/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_15/dense_139/Tensordot/stack?
+sequential_15/dense_139/Tensordot/transpose	Transpose*layer_normalization_30/batchnorm/add_1:z:01sequential_15/dense_139/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2-
+sequential_15/dense_139/Tensordot/transpose?
)sequential_15/dense_139/Tensordot/ReshapeReshape/sequential_15/dense_139/Tensordot/transpose:y:00sequential_15/dense_139/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)sequential_15/dense_139/Tensordot/Reshape?
(sequential_15/dense_139/Tensordot/MatMulMatMul2sequential_15/dense_139/Tensordot/Reshape:output:08sequential_15/dense_139/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(sequential_15/dense_139/Tensordot/MatMul?
)sequential_15/dense_139/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_139/Tensordot/Const_2?
/sequential_15/dense_139/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_139/Tensordot/concat_1/axis?
*sequential_15/dense_139/Tensordot/concat_1ConcatV23sequential_15/dense_139/Tensordot/GatherV2:output:02sequential_15/dense_139/Tensordot/Const_2:output:08sequential_15/dense_139/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*sequential_15/dense_139/Tensordot/concat_1?
!sequential_15/dense_139/TensordotReshape2sequential_15/dense_139/Tensordot/MatMul:product:03sequential_15/dense_139/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2#
!sequential_15/dense_139/Tensordot?
.sequential_15/dense_139/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_dense_139_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/dense_139/BiasAdd/ReadVariableOp?
sequential_15/dense_139/BiasAddBiasAdd*sequential_15/dense_139/Tensordot:output:06sequential_15/dense_139/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2!
sequential_15/dense_139/BiasAdd?
sequential_15/dense_139/ReluRelu(sequential_15/dense_139/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_15/dense_139/Relu?
0sequential_15/dense_140/Tensordot/ReadVariableOpReadVariableOp9sequential_15_dense_140_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype022
0sequential_15/dense_140/Tensordot/ReadVariableOp?
&sequential_15/dense_140/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_15/dense_140/Tensordot/axes?
&sequential_15/dense_140/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2(
&sequential_15/dense_140/Tensordot/free?
'sequential_15/dense_140/Tensordot/ShapeShape*sequential_15/dense_139/Relu:activations:0*
T0*
_output_shapes
:2)
'sequential_15/dense_140/Tensordot/Shape?
/sequential_15/dense_140/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_140/Tensordot/GatherV2/axis?
*sequential_15/dense_140/Tensordot/GatherV2GatherV20sequential_15/dense_140/Tensordot/Shape:output:0/sequential_15/dense_140/Tensordot/free:output:08sequential_15/dense_140/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_15/dense_140/Tensordot/GatherV2?
1sequential_15/dense_140/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_15/dense_140/Tensordot/GatherV2_1/axis?
,sequential_15/dense_140/Tensordot/GatherV2_1GatherV20sequential_15/dense_140/Tensordot/Shape:output:0/sequential_15/dense_140/Tensordot/axes:output:0:sequential_15/dense_140/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2.
,sequential_15/dense_140/Tensordot/GatherV2_1?
'sequential_15/dense_140/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_15/dense_140/Tensordot/Const?
&sequential_15/dense_140/Tensordot/ProdProd3sequential_15/dense_140/Tensordot/GatherV2:output:00sequential_15/dense_140/Tensordot/Const:output:0*
T0*
_output_shapes
: 2(
&sequential_15/dense_140/Tensordot/Prod?
)sequential_15/dense_140/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_140/Tensordot/Const_1?
(sequential_15/dense_140/Tensordot/Prod_1Prod5sequential_15/dense_140/Tensordot/GatherV2_1:output:02sequential_15/dense_140/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2*
(sequential_15/dense_140/Tensordot/Prod_1?
-sequential_15/dense_140/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_15/dense_140/Tensordot/concat/axis?
(sequential_15/dense_140/Tensordot/concatConcatV2/sequential_15/dense_140/Tensordot/free:output:0/sequential_15/dense_140/Tensordot/axes:output:06sequential_15/dense_140/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_15/dense_140/Tensordot/concat?
'sequential_15/dense_140/Tensordot/stackPack/sequential_15/dense_140/Tensordot/Prod:output:01sequential_15/dense_140/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2)
'sequential_15/dense_140/Tensordot/stack?
+sequential_15/dense_140/Tensordot/transpose	Transpose*sequential_15/dense_139/Relu:activations:01sequential_15/dense_140/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2-
+sequential_15/dense_140/Tensordot/transpose?
)sequential_15/dense_140/Tensordot/ReshapeReshape/sequential_15/dense_140/Tensordot/transpose:y:00sequential_15/dense_140/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2+
)sequential_15/dense_140/Tensordot/Reshape?
(sequential_15/dense_140/Tensordot/MatMulMatMul2sequential_15/dense_140/Tensordot/Reshape:output:08sequential_15/dense_140/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(sequential_15/dense_140/Tensordot/MatMul?
)sequential_15/dense_140/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_15/dense_140/Tensordot/Const_2?
/sequential_15/dense_140/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_15/dense_140/Tensordot/concat_1/axis?
*sequential_15/dense_140/Tensordot/concat_1ConcatV23sequential_15/dense_140/Tensordot/GatherV2:output:02sequential_15/dense_140/Tensordot/Const_2:output:08sequential_15/dense_140/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2,
*sequential_15/dense_140/Tensordot/concat_1?
!sequential_15/dense_140/TensordotReshape2sequential_15/dense_140/Tensordot/MatMul:product:03sequential_15/dense_140/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2#
!sequential_15/dense_140/Tensordot?
.sequential_15/dense_140/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_dense_140_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_15/dense_140/BiasAdd/ReadVariableOp?
sequential_15/dense_140/BiasAddBiasAdd*sequential_15/dense_140/Tensordot:output:06sequential_15/dense_140/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2!
sequential_15/dense_140/BiasAdd?
dropout_31/IdentityIdentity(sequential_15/dense_140/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_31/Identity?
add_1AddV2*layer_normalization_30/batchnorm/add_1:z:0dropout_31/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add_1?
5layer_normalization_31/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_31/moments/mean/reduction_indices?
#layer_normalization_31/moments/meanMean	add_1:z:0>layer_normalization_31/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_31/moments/mean?
+layer_normalization_31/moments/StopGradientStopGradient,layer_normalization_31/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_31/moments/StopGradient?
0layer_normalization_31/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_31/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_31/moments/SquaredDifference?
9layer_normalization_31/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_31/moments/variance/reduction_indices?
'layer_normalization_31/moments/varianceMean4layer_normalization_31/moments/SquaredDifference:z:0Blayer_normalization_31/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_31/moments/variance?
&layer_normalization_31/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_31/batchnorm/add/y?
$layer_normalization_31/batchnorm/addAddV20layer_normalization_31/moments/variance:output:0/layer_normalization_31/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_31/batchnorm/add?
&layer_normalization_31/batchnorm/RsqrtRsqrt(layer_normalization_31/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_31/batchnorm/Rsqrt?
3layer_normalization_31/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_31_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_31/batchnorm/mul/ReadVariableOp?
$layer_normalization_31/batchnorm/mulMul*layer_normalization_31/batchnorm/Rsqrt:y:0;layer_normalization_31/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_31/batchnorm/mul?
&layer_normalization_31/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/mul_1?
&layer_normalization_31/batchnorm/mul_2Mul,layer_normalization_31/moments/mean:output:0(layer_normalization_31/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/mul_2?
/layer_normalization_31/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_31_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_31/batchnorm/ReadVariableOp?
$layer_normalization_31/batchnorm/subSub7layer_normalization_31/batchnorm/ReadVariableOp:value:0*layer_normalization_31/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_31/batchnorm/sub?
&layer_normalization_31/batchnorm/add_1AddV2*layer_normalization_31/batchnorm/mul_1:z:0(layer_normalization_31/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_31/batchnorm/add_1?
IdentityIdentity*layer_normalization_31/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp0^layer_normalization_30/batchnorm/ReadVariableOp4^layer_normalization_30/batchnorm/mul/ReadVariableOp0^layer_normalization_31/batchnorm/ReadVariableOp4^layer_normalization_31/batchnorm/mul/ReadVariableOp>^multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp>^multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp@^multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp/^sequential_15/dense_139/BiasAdd/ReadVariableOp1^sequential_15/dense_139/Tensordot/ReadVariableOp/^sequential_15/dense_140/BiasAdd/ReadVariableOp1^sequential_15/dense_140/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2b
/layer_normalization_30/batchnorm/ReadVariableOp/layer_normalization_30/batchnorm/ReadVariableOp2j
3layer_normalization_30/batchnorm/mul/ReadVariableOp3layer_normalization_30/batchnorm/mul/ReadVariableOp2b
/layer_normalization_31/batchnorm/ReadVariableOp/layer_normalization_31/batchnorm/ReadVariableOp2j
3layer_normalization_31/batchnorm/mul/ReadVariableOp3layer_normalization_31/batchnorm/mul/ReadVariableOp2~
=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_135/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_135/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_136/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_136/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_137/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_137/Tensordot/ReadVariableOp2~
=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp=multi_head_self_attention_15/dense_138/BiasAdd/ReadVariableOp2?
?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp?multi_head_self_attention_15/dense_138/Tensordot/ReadVariableOp2`
.sequential_15/dense_139/BiasAdd/ReadVariableOp.sequential_15/dense_139/BiasAdd/ReadVariableOp2d
0sequential_15/dense_139/Tensordot/ReadVariableOp0sequential_15/dense_139/Tensordot/ReadVariableOp2`
.sequential_15/dense_140/BiasAdd/ReadVariableOp.sequential_15/dense_140/BiasAdd/ReadVariableOp2d
0sequential_15/dense_140/Tensordot/ReadVariableOp0sequential_15/dense_140/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_6976554
aaindex_input
	aux_input
input_16
unknown:( 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7:  
	unknown_8: 
	unknown_9: 

unknown_10: 

unknown_11:  

unknown_12: 

unknown_13:  

unknown_14: 

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18:

unknown_19:@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_16	aux_inputaaindex_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_69749632
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapess
q:?????????:?????????:?????????(: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_nameaaindex_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input:QM
'
_output_shapes
:?????????(
"
_user_specified_name
input_16
?
?
G__inference_aux_output_layer_call_and_return_conditional_losses_6977952

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_6975044

inputs#
dense_139_6975002:  
dense_139_6975004: #
dense_140_6975038:  
dense_140_6975040: 
identity??!dense_139/StatefulPartitionedCall?!dense_140/StatefulPartitionedCall?
!dense_139/StatefulPartitionedCallStatefulPartitionedCallinputsdense_139_6975002dense_139_6975004*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_139_layer_call_and_return_conditional_losses_69750012#
!dense_139/StatefulPartitionedCall?
!dense_140/StatefulPartitionedCallStatefulPartitionedCall*dense_139/StatefulPartitionedCall:output:0dense_140_6975038dense_140_6975040*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_140_layer_call_and_return_conditional_losses_69750372#
!dense_140/StatefulPartitionedCall?
IdentityIdentity*dense_140/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp"^dense_139/StatefulPartitionedCall"^dense_140/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2F
!dense_139/StatefulPartitionedCall!dense_139/StatefulPartitionedCall2F
!dense_140/StatefulPartitionedCall!dense_140/StatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?!
?
F__inference_dense_139_layer_call_and_return_conditional_losses_6975001

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????( : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
Y
=__inference_global_average_pooling1d_15_layer_call_fn_6977941

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *a
f\RZ
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_69755042
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????( :S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
+__inference_dense_143_layer_call_fn_6978036

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_143_layer_call_and_return_conditional_losses_69755782
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
j
0__inference_concatenate_15_layer_call_fn_6977976
inputs_0
inputs_1
inputs_2
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concatenate_15_layer_call_and_return_conditional_losses_69755312
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
aaindex_input6
serving_default_aaindex_input:0?????????
?
	aux_input2
serving_default_aux_input:0?????????
=
input_161
serving_default_input_16:0?????????(>

aux_output0
StatefulPartitionedCall:0??????????
main_output0
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?
	token_emb
pos_emb
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
regularization_losses
 trainable_variables
!	variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
#regularization_losses
$trainable_variables
%	variables
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
-regularization_losses
.trainable_variables
/	variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Iiter

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_rate'm?(m?1m?2m?7m?8m?=m?>m?Cm?Dm?Nm?Om?Pm?Qm?Rm?Sm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?]m?^m?_m?'v?(v?1v?2v?7v?8v?=v?>v?Cv?Dv?Nv?Ov?Pv?Qv?Rv?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?]v?^v?_v?"
	optimizer
 "
trackable_list_wrapper
?
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
Z12
[13
\14
]15
^16
_17
'18
(19
120
221
722
823
=24
>25
C26
D27"
trackable_list_wrapper
?
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
Z12
[13
\14
]15
^16
_17
'18
(19
120
221
722
823
=24
>25
C26
D27"
trackable_list_wrapper
?
regularization_losses
trainable_variables
`layer_metrics
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
N
embeddings
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
O
embeddings
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
?
regularization_losses
trainable_variables
mlayer_metrics

nlayers
onon_trainable_variables
pmetrics
qlayer_regularization_losses
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
rquery_dense
s	key_dense
tvalue_dense
ucombine_heads
vregularization_losses
wtrainable_variables
x	variables
y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
zlayer_with_weights-0
zlayer-0
{layer_with_weights-1
{layer-1
|regularization_losses
}trainable_variables
~	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
	?axis
	\gamma
]beta
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis
	^gamma
_beta
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
?
P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
Z10
[11
\12
]13
^14
_15"
trackable_list_wrapper
?
P0
Q1
R2
S3
T4
U5
V6
W7
X8
Y9
Z10
[11
\12
]13
^14
_15"
trackable_list_wrapper
?
regularization_losses
 trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
!	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
#regularization_losses
$trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
%	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:! 2aux_output/kernel
:2aux_output/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
)regularization_losses
*trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
+	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-regularization_losses
.trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
/	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": @2dense_141/kernel
:@2dense_141/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
3regularization_losses
4trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
5	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": @@2dense_142/kernel
:@2dense_142/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
9regularization_losses
:trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
;	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": @@2dense_143/kernel
:@2dense_143/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?regularization_losses
@trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
A	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"@2main_output/kernel
:2main_output/bias
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
?
Eregularization_losses
Ftrainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
G	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
I:G 27token_and_position_embedding_15/embedding_30/embeddings
I:G( 27token_and_position_embedding_15/embedding_31/embeddings
T:R  2Btransformer_block_15/multi_head_self_attention_15/dense_135/kernel
N:L 2@transformer_block_15/multi_head_self_attention_15/dense_135/bias
T:R  2Btransformer_block_15/multi_head_self_attention_15/dense_136/kernel
N:L 2@transformer_block_15/multi_head_self_attention_15/dense_136/bias
T:R  2Btransformer_block_15/multi_head_self_attention_15/dense_137/kernel
N:L 2@transformer_block_15/multi_head_self_attention_15/dense_137/bias
T:R  2Btransformer_block_15/multi_head_self_attention_15/dense_138/kernel
N:L 2@transformer_block_15/multi_head_self_attention_15/dense_138/bias
":   2dense_139/kernel
: 2dense_139/bias
":   2dense_140/kernel
: 2dense_140/bias
?:= 21transformer_block_15/layer_normalization_30/gamma
>:< 20transformer_block_15/layer_normalization_30/beta
?:= 21transformer_block_15/layer_normalization_31/gamma
>:< 20transformer_block_15/layer_normalization_31/beta
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
H
?0
?1
?2
?3
?4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
?
eregularization_losses
ftrainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
g	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
O0"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
?
iregularization_losses
jtrainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
k	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Pkernel
Qbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Rkernel
Sbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Tkernel
Ubias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Vkernel
Wbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
X
P0
Q1
R2
S3
T4
U5
V6
W7"
trackable_list_wrapper
X
P0
Q1
R2
S3
T4
U5
V6
W7"
trackable_list_wrapper
?
vregularization_losses
wtrainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
x	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Xkernel
Ybias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Zkernel
[bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
 "
trackable_list_wrapper
<
X0
Y1
Z2
[3"
trackable_list_wrapper
<
X0
Y1
Z2
[3"
trackable_list_wrapper
?
|regularization_losses
}trainable_variables
?layer_metrics
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
~	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
<
r0
s1
t2
u3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
?
?regularization_losses
?trainable_variables
?layer_metrics
?layers
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(:& 2Adam/aux_output/kernel/m
": 2Adam/aux_output/bias/m
':%@2Adam/dense_141/kernel/m
!:@2Adam/dense_141/bias/m
':%@@2Adam/dense_142/kernel/m
!:@2Adam/dense_142/bias/m
':%@@2Adam/dense_143/kernel/m
!:@2Adam/dense_143/bias/m
):'@2Adam/main_output/kernel/m
#:!2Adam/main_output/bias/m
N:L 2>Adam/token_and_position_embedding_15/embedding_30/embeddings/m
N:L( 2>Adam/token_and_position_embedding_15/embedding_31/embeddings/m
Y:W  2IAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/m
S:Q 2GAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/m
Y:W  2IAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/m
S:Q 2GAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/m
Y:W  2IAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/m
S:Q 2GAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/m
Y:W  2IAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/m
S:Q 2GAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/m
':%  2Adam/dense_139/kernel/m
!: 2Adam/dense_139/bias/m
':%  2Adam/dense_140/kernel/m
!: 2Adam/dense_140/bias/m
D:B 28Adam/transformer_block_15/layer_normalization_30/gamma/m
C:A 27Adam/transformer_block_15/layer_normalization_30/beta/m
D:B 28Adam/transformer_block_15/layer_normalization_31/gamma/m
C:A 27Adam/transformer_block_15/layer_normalization_31/beta/m
(:& 2Adam/aux_output/kernel/v
": 2Adam/aux_output/bias/v
':%@2Adam/dense_141/kernel/v
!:@2Adam/dense_141/bias/v
':%@@2Adam/dense_142/kernel/v
!:@2Adam/dense_142/bias/v
':%@@2Adam/dense_143/kernel/v
!:@2Adam/dense_143/bias/v
):'@2Adam/main_output/kernel/v
#:!2Adam/main_output/bias/v
N:L 2>Adam/token_and_position_embedding_15/embedding_30/embeddings/v
N:L( 2>Adam/token_and_position_embedding_15/embedding_31/embeddings/v
Y:W  2IAdam/transformer_block_15/multi_head_self_attention_15/dense_135/kernel/v
S:Q 2GAdam/transformer_block_15/multi_head_self_attention_15/dense_135/bias/v
Y:W  2IAdam/transformer_block_15/multi_head_self_attention_15/dense_136/kernel/v
S:Q 2GAdam/transformer_block_15/multi_head_self_attention_15/dense_136/bias/v
Y:W  2IAdam/transformer_block_15/multi_head_self_attention_15/dense_137/kernel/v
S:Q 2GAdam/transformer_block_15/multi_head_self_attention_15/dense_137/bias/v
Y:W  2IAdam/transformer_block_15/multi_head_self_attention_15/dense_138/kernel/v
S:Q 2GAdam/transformer_block_15/multi_head_self_attention_15/dense_138/bias/v
':%  2Adam/dense_139/kernel/v
!: 2Adam/dense_139/bias/v
':%  2Adam/dense_140/kernel/v
!: 2Adam/dense_140/bias/v
D:B 28Adam/transformer_block_15/layer_normalization_30/gamma/v
C:A 27Adam/transformer_block_15/layer_normalization_30/beta/v
D:B 28Adam/transformer_block_15/layer_normalization_31/gamma/v
C:A 27Adam/transformer_block_15/layer_normalization_31/beta/v
?2?
E__inference_model_15_layer_call_and_return_conditional_losses_6976860
E__inference_model_15_layer_call_and_return_conditional_losses_6977180
E__inference_model_15_layer_call_and_return_conditional_losses_6976409
E__inference_model_15_layer_call_and_return_conditional_losses_6976481?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_model_15_layer_call_fn_6975664
*__inference_model_15_layer_call_fn_6977245
*__inference_model_15_layer_call_fn_6977310
*__inference_model_15_layer_call_fn_6976337?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_6974963input_16	aux_inputaaindex_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
\__inference_token_and_position_embedding_15_layer_call_and_return_conditional_losses_6977334?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_token_and_position_embedding_15_layer_call_fn_6977343?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_6977587
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_6977845?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
6__inference_transformer_block_15_layer_call_fn_6977882
6__inference_transformer_block_15_layer_call_fn_6977919?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_6977925
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_6977931?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
=__inference_global_average_pooling1d_15_layer_call_fn_6977936
=__inference_global_average_pooling1d_15_layer_call_fn_6977941?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_aux_output_layer_call_and_return_conditional_losses_6977952?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_aux_output_layer_call_fn_6977961?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_concatenate_15_layer_call_and_return_conditional_losses_6977969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_concatenate_15_layer_call_fn_6977976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_141_layer_call_and_return_conditional_losses_6977987?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_141_layer_call_fn_6977996?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_142_layer_call_and_return_conditional_losses_6978007?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_142_layer_call_fn_6978016?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_143_layer_call_and_return_conditional_losses_6978027?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_143_layer_call_fn_6978036?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_main_output_layer_call_and_return_conditional_losses_6978047?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_main_output_layer_call_fn_6978056?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_6976554aaindex_input	aux_inputinput_16"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_sequential_15_layer_call_and_return_conditional_losses_6978113
J__inference_sequential_15_layer_call_and_return_conditional_losses_6978170
J__inference_sequential_15_layer_call_and_return_conditional_losses_6975142
J__inference_sequential_15_layer_call_and_return_conditional_losses_6975156?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_sequential_15_layer_call_fn_6975055
/__inference_sequential_15_layer_call_fn_6978183
/__inference_sequential_15_layer_call_fn_6978196
/__inference_sequential_15_layer_call_fn_6975128?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_139_layer_call_and_return_conditional_losses_6978227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_139_layer_call_fn_6978236?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_140_layer_call_and_return_conditional_losses_6978266?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_140_layer_call_fn_6978275?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_6974963?ONPQRSTUVW\]XYZ[^_'(1278=>CD???
z?w
u?r
"?
input_16?????????(
#? 
	aux_input?????????
'?$
aaindex_input?????????
? "m?j
2

aux_output$?!

aux_output?????????
4
main_output%?"
main_output??????????
G__inference_aux_output_layer_call_and_return_conditional_losses_6977952\'(/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? 
,__inference_aux_output_layer_call_fn_6977961O'(/?,
%?"
 ?
inputs????????? 
? "???????????
K__inference_concatenate_15_layer_call_and_return_conditional_losses_6977969?~?{
t?q
o?l
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
? "%?"
?
0?????????
? ?
0__inference_concatenate_15_layer_call_fn_6977976?~?{
t?q
o?l
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
? "???????????
F__inference_dense_139_layer_call_and_return_conditional_losses_6978227dXY3?0
)?&
$?!
inputs?????????( 
? ")?&
?
0?????????( 
? ?
+__inference_dense_139_layer_call_fn_6978236WXY3?0
)?&
$?!
inputs?????????( 
? "??????????( ?
F__inference_dense_140_layer_call_and_return_conditional_losses_6978266dZ[3?0
)?&
$?!
inputs?????????( 
? ")?&
?
0?????????( 
? ?
+__inference_dense_140_layer_call_fn_6978275WZ[3?0
)?&
$?!
inputs?????????( 
? "??????????( ?
F__inference_dense_141_layer_call_and_return_conditional_losses_6977987\12/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? ~
+__inference_dense_141_layer_call_fn_6977996O12/?,
%?"
 ?
inputs?????????
? "??????????@?
F__inference_dense_142_layer_call_and_return_conditional_losses_6978007\78/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
+__inference_dense_142_layer_call_fn_6978016O78/?,
%?"
 ?
inputs?????????@
? "??????????@?
F__inference_dense_143_layer_call_and_return_conditional_losses_6978027\=>/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ~
+__inference_dense_143_layer_call_fn_6978036O=>/?,
%?"
 ?
inputs?????????@
? "??????????@?
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_6977925{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
X__inference_global_average_pooling1d_15_layer_call_and_return_conditional_losses_6977931`7?4
-?*
$?!
inputs?????????( 

 
? "%?"
?
0????????? 
? ?
=__inference_global_average_pooling1d_15_layer_call_fn_6977936nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
=__inference_global_average_pooling1d_15_layer_call_fn_6977941S7?4
-?*
$?!
inputs?????????( 

 
? "?????????? ?
H__inference_main_output_layer_call_and_return_conditional_losses_6978047\CD/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ?
-__inference_main_output_layer_call_fn_6978056OCD/?,
%?"
 ?
inputs?????????@
? "???????????
E__inference_model_15_layer_call_and_return_conditional_losses_6976409?ONPQRSTUVW\]XYZ[^_'(1278=>CD???
??
u?r
"?
input_16?????????(
#? 
	aux_input?????????
'?$
aaindex_input?????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
E__inference_model_15_layer_call_and_return_conditional_losses_6976481?ONPQRSTUVW\]XYZ[^_'(1278=>CD???
??
u?r
"?
input_16?????????(
#? 
	aux_input?????????
'?$
aaindex_input?????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
E__inference_model_15_layer_call_and_return_conditional_losses_6976860?ONPQRSTUVW\]XYZ[^_'(1278=>CD???
|?y
o?l
"?
inputs/0?????????(
"?
inputs/1?????????
"?
inputs/2?????????
p 

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
E__inference_model_15_layer_call_and_return_conditional_losses_6977180?ONPQRSTUVW\]XYZ[^_'(1278=>CD???
|?y
o?l
"?
inputs/0?????????(
"?
inputs/1?????????
"?
inputs/2?????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
*__inference_model_15_layer_call_fn_6975664?ONPQRSTUVW\]XYZ[^_'(1278=>CD???
??
u?r
"?
input_16?????????(
#? 
	aux_input?????????
'?$
aaindex_input?????????
p 

 
? "=?:
?
0?????????
?
1??????????
*__inference_model_15_layer_call_fn_6976337?ONPQRSTUVW\]XYZ[^_'(1278=>CD???
??
u?r
"?
input_16?????????(
#? 
	aux_input?????????
'?$
aaindex_input?????????
p

 
? "=?:
?
0?????????
?
1??????????
*__inference_model_15_layer_call_fn_6977245?ONPQRSTUVW\]XYZ[^_'(1278=>CD???
|?y
o?l
"?
inputs/0?????????(
"?
inputs/1?????????
"?
inputs/2?????????
p 

 
? "=?:
?
0?????????
?
1??????????
*__inference_model_15_layer_call_fn_6977310?ONPQRSTUVW\]XYZ[^_'(1278=>CD???
|?y
o?l
"?
inputs/0?????????(
"?
inputs/1?????????
"?
inputs/2?????????
p

 
? "=?:
?
0?????????
?
1??????????
J__inference_sequential_15_layer_call_and_return_conditional_losses_6975142wXYZ[D?A
:?7
-?*
dense_139_input?????????( 
p 

 
? ")?&
?
0?????????( 
? ?
J__inference_sequential_15_layer_call_and_return_conditional_losses_6975156wXYZ[D?A
:?7
-?*
dense_139_input?????????( 
p

 
? ")?&
?
0?????????( 
? ?
J__inference_sequential_15_layer_call_and_return_conditional_losses_6978113nXYZ[;?8
1?.
$?!
inputs?????????( 
p 

 
? ")?&
?
0?????????( 
? ?
J__inference_sequential_15_layer_call_and_return_conditional_losses_6978170nXYZ[;?8
1?.
$?!
inputs?????????( 
p

 
? ")?&
?
0?????????( 
? ?
/__inference_sequential_15_layer_call_fn_6975055jXYZ[D?A
:?7
-?*
dense_139_input?????????( 
p 

 
? "??????????( ?
/__inference_sequential_15_layer_call_fn_6975128jXYZ[D?A
:?7
-?*
dense_139_input?????????( 
p

 
? "??????????( ?
/__inference_sequential_15_layer_call_fn_6978183aXYZ[;?8
1?.
$?!
inputs?????????( 
p 

 
? "??????????( ?
/__inference_sequential_15_layer_call_fn_6978196aXYZ[;?8
1?.
$?!
inputs?????????( 
p

 
? "??????????( ?
%__inference_signature_wrapper_6976554?ONPQRSTUVW\]XYZ[^_'(1278=>CD???
? 
???
8
aaindex_input'?$
aaindex_input?????????
0
	aux_input#? 
	aux_input?????????
.
input_16"?
input_16?????????("m?j
2

aux_output$?!

aux_output?????????
4
main_output%?"
main_output??????????
\__inference_token_and_position_embedding_15_layer_call_and_return_conditional_losses_6977334[ON*?'
 ?
?
x?????????(
? ")?&
?
0?????????( 
? ?
A__inference_token_and_position_embedding_15_layer_call_fn_6977343NON*?'
 ?
?
x?????????(
? "??????????( ?
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_6977587vPQRSTUVW\]XYZ[^_7?4
-?*
$?!
inputs?????????( 
p 
? ")?&
?
0?????????( 
? ?
Q__inference_transformer_block_15_layer_call_and_return_conditional_losses_6977845vPQRSTUVW\]XYZ[^_7?4
-?*
$?!
inputs?????????( 
p
? ")?&
?
0?????????( 
? ?
6__inference_transformer_block_15_layer_call_fn_6977882iPQRSTUVW\]XYZ[^_7?4
-?*
$?!
inputs?????????( 
p 
? "??????????( ?
6__inference_transformer_block_15_layer_call_fn_6977919iPQRSTUVW\]XYZ[^_7?4
-?*
$?!
inputs?????????( 
p
? "??????????( 