??3
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
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??0
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
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

:@*
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:@*
dtype0
z
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:@@*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:@*
dtype0
z
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:@@*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
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
5token_and_position_embedding_2/embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75token_and_position_embedding_2/embedding_4/embeddings
?
Itoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_4/embeddings*
_output_shapes

: *
dtype0
?
5token_and_position_embedding_2/embedding_5/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *F
shared_name75token_and_position_embedding_2/embedding_5/embeddings
?
Itoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpReadVariableOp5token_and_position_embedding_2/embedding_5/embeddings*
_output_shapes

:( *
dtype0
?
?transformer_block_2/multi_head_self_attention_2/dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *P
shared_nameA?transformer_block_2/multi_head_self_attention_2/dense_18/kernel
?
Stransformer_block_2/multi_head_self_attention_2/dense_18/kernel/Read/ReadVariableOpReadVariableOp?transformer_block_2/multi_head_self_attention_2/dense_18/kernel*
_output_shapes

:  *
dtype0
?
=transformer_block_2/multi_head_self_attention_2/dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=transformer_block_2/multi_head_self_attention_2/dense_18/bias
?
Qtransformer_block_2/multi_head_self_attention_2/dense_18/bias/Read/ReadVariableOpReadVariableOp=transformer_block_2/multi_head_self_attention_2/dense_18/bias*
_output_shapes
: *
dtype0
?
?transformer_block_2/multi_head_self_attention_2/dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *P
shared_nameA?transformer_block_2/multi_head_self_attention_2/dense_19/kernel
?
Stransformer_block_2/multi_head_self_attention_2/dense_19/kernel/Read/ReadVariableOpReadVariableOp?transformer_block_2/multi_head_self_attention_2/dense_19/kernel*
_output_shapes

:  *
dtype0
?
=transformer_block_2/multi_head_self_attention_2/dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=transformer_block_2/multi_head_self_attention_2/dense_19/bias
?
Qtransformer_block_2/multi_head_self_attention_2/dense_19/bias/Read/ReadVariableOpReadVariableOp=transformer_block_2/multi_head_self_attention_2/dense_19/bias*
_output_shapes
: *
dtype0
?
?transformer_block_2/multi_head_self_attention_2/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *P
shared_nameA?transformer_block_2/multi_head_self_attention_2/dense_20/kernel
?
Stransformer_block_2/multi_head_self_attention_2/dense_20/kernel/Read/ReadVariableOpReadVariableOp?transformer_block_2/multi_head_self_attention_2/dense_20/kernel*
_output_shapes

:  *
dtype0
?
=transformer_block_2/multi_head_self_attention_2/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=transformer_block_2/multi_head_self_attention_2/dense_20/bias
?
Qtransformer_block_2/multi_head_self_attention_2/dense_20/bias/Read/ReadVariableOpReadVariableOp=transformer_block_2/multi_head_self_attention_2/dense_20/bias*
_output_shapes
: *
dtype0
?
?transformer_block_2/multi_head_self_attention_2/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *P
shared_nameA?transformer_block_2/multi_head_self_attention_2/dense_21/kernel
?
Stransformer_block_2/multi_head_self_attention_2/dense_21/kernel/Read/ReadVariableOpReadVariableOp?transformer_block_2/multi_head_self_attention_2/dense_21/kernel*
_output_shapes

:  *
dtype0
?
=transformer_block_2/multi_head_self_attention_2/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=transformer_block_2/multi_head_self_attention_2/dense_21/bias
?
Qtransformer_block_2/multi_head_self_attention_2/dense_21/bias/Read/ReadVariableOpReadVariableOp=transformer_block_2/multi_head_self_attention_2/dense_21/bias*
_output_shapes
: *
dtype0
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

:  *
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes
: *
dtype0
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:  *
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
: *
dtype0
?
/transformer_block_2/layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_2/layer_normalization_4/gamma
?
Ctransformer_block_2/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_2/layer_normalization_4/gamma*
_output_shapes
: *
dtype0
?
.transformer_block_2/layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_2/layer_normalization_4/beta
?
Btransformer_block_2/layer_normalization_4/beta/Read/ReadVariableOpReadVariableOp.transformer_block_2/layer_normalization_4/beta*
_output_shapes
: *
dtype0
?
/transformer_block_2/layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_2/layer_normalization_5/gamma
?
Ctransformer_block_2/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOp/transformer_block_2/layer_normalization_5/gamma*
_output_shapes
: *
dtype0
?
.transformer_block_2/layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.transformer_block_2/layer_normalization_5/beta
?
Btransformer_block_2/layer_normalization_5/beta/Read/ReadVariableOpReadVariableOp.transformer_block_2/layer_normalization_5/beta*
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
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_24/kernel/m
?
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_25/kernel/m
?
*Adam/dense_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_25/bias/m
y
(Adam/dense_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_26/kernel/m
?
*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_26/bias/m
y
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
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
<Adam/token_and_position_embedding_2/embedding_4/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/token_and_position_embedding_2/embedding_4/embeddings/m
?
PAdam/token_and_position_embedding_2/embedding_4/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_4/embeddings/m*
_output_shapes

: *
dtype0
?
<Adam/token_and_position_embedding_2/embedding_5/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *M
shared_name><Adam/token_and_position_embedding_2/embedding_5/embeddings/m
?
PAdam/token_and_position_embedding_2/embedding_5/embeddings/m/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_5/embeddings/m*
_output_shapes

:( *
dtype0
?
FAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/m
?
ZAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/m*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/m
?
XAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/m/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/m*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/m
?
ZAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/m*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/m
?
XAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/m/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/m*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/m
?
ZAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/m*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/m
?
XAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/m/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/m*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/m
?
ZAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/m*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/m
?
XAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/m/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_22/kernel/m
?
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes

:  *
dtype0
?
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_23/kernel/m
?
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes

:  *
dtype0
?
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
: *
dtype0
?
6Adam/transformer_block_2/layer_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_4/gamma/m
?
JAdam/transformer_block_2/layer_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_4/gamma/m*
_output_shapes
: *
dtype0
?
5Adam/transformer_block_2/layer_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_4/beta/m
?
IAdam/transformer_block_2/layer_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_4/beta/m*
_output_shapes
: *
dtype0
?
6Adam/transformer_block_2/layer_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_5/gamma/m
?
JAdam/transformer_block_2/layer_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_5/gamma/m*
_output_shapes
: *
dtype0
?
5Adam/transformer_block_2/layer_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_5/beta/m
?
IAdam/transformer_block_2/layer_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_5/beta/m*
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
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_24/kernel/v
?
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_25/kernel/v
?
*Adam/dense_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_25/bias/v
y
(Adam/dense_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_25/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_26/kernel/v
?
*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_26/bias/v
y
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
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
<Adam/token_and_position_embedding_2/embedding_4/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><Adam/token_and_position_embedding_2/embedding_4/embeddings/v
?
PAdam/token_and_position_embedding_2/embedding_4/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_4/embeddings/v*
_output_shapes

: *
dtype0
?
<Adam/token_and_position_embedding_2/embedding_5/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *M
shared_name><Adam/token_and_position_embedding_2/embedding_5/embeddings/v
?
PAdam/token_and_position_embedding_2/embedding_5/embeddings/v/Read/ReadVariableOpReadVariableOp<Adam/token_and_position_embedding_2/embedding_5/embeddings/v*
_output_shapes

:( *
dtype0
?
FAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/v
?
ZAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/v*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/v
?
XAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/v/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/v*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/v
?
ZAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/v*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/v
?
XAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/v/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/v*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/v
?
ZAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/v*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/v
?
XAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/v/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/v*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/v
?
ZAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/v*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/v
?
XAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/v/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_22/kernel/v
?
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes

:  *
dtype0
?
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_23/kernel/v
?
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes

:  *
dtype0
?
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
: *
dtype0
?
6Adam/transformer_block_2/layer_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_4/gamma/v
?
JAdam/transformer_block_2/layer_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_4/gamma/v*
_output_shapes
: *
dtype0
?
5Adam/transformer_block_2/layer_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_4/beta/v
?
IAdam/transformer_block_2/layer_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_4/beta/v*
_output_shapes
: *
dtype0
?
6Adam/transformer_block_2/layer_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_2/layer_normalization_5/gamma/v
?
JAdam/transformer_block_2/layer_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_2/layer_normalization_5/gamma/v*
_output_shapes
: *
dtype0
?
5Adam/transformer_block_2/layer_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *F
shared_name75Adam/transformer_block_2/layer_normalization_5/beta/v
?
IAdam/transformer_block_2/layer_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp5Adam/transformer_block_2/layer_normalization_5/beta/v*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
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
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
n
	token_emb
pos_emb
regularization_losses
	variables
trainable_variables
	keras_api
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
 
R
,regularization_losses
-	variables
.trainable_variables
/	keras_api
h

0kernel
1bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
h

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
h

Bkernel
Cbias
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
?
Hiter

Ibeta_1

Jbeta_2
	Kdecay
Llearning_rate&m?'m?0m?1m?6m?7m?<m?=m?Bm?Cm?Mm?Nm?Om?Pm?Qm?Rm?Sm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?]m?^m?&v?'v?0v?1v?6v?7v?<v?=v?Bv?Cv?Mv?Nv?Ov?Pv?Qv?Rv?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?]v?^v?
 
?
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
&18
'19
020
121
622
723
<24
=25
B26
C27
?
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
&18
'19
020
121
622
723
<24
=25
B26
C27
?
_metrics
`layer_metrics
regularization_losses
alayer_regularization_losses
	variables
bnon_trainable_variables

clayers
trainable_variables
 
b
M
embeddings
dregularization_losses
e	variables
ftrainable_variables
g	keras_api
b
N
embeddings
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
 

M0
N1

M0
N1
?
lmetrics
mlayer_metrics
regularization_losses
nlayer_regularization_losses
	variables
onon_trainable_variables

players
trainable_variables
?
qquery_dense
r	key_dense
svalue_dense
tcombine_heads
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
?
ylayer_with_weights-0
ylayer-0
zlayer_with_weights-1
zlayer-1
{regularization_losses
|	variables
}trainable_variables
~	keras_api
u
axis
	[gamma
\beta
?regularization_losses
?	variables
?trainable_variables
?	keras_api
v
	?axis
	]gamma
^beta
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
V
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
v
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15
v
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15
?
?metrics
?layer_metrics
regularization_losses
 ?layer_regularization_losses
	variables
?non_trainable_variables
?layers
 trainable_variables
 
 
 
?
?metrics
?layer_metrics
"regularization_losses
 ?layer_regularization_losses
#	variables
?non_trainable_variables
?layers
$trainable_variables
][
VARIABLE_VALUEaux_output/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEaux_output/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

&0
'1

&0
'1
?
?metrics
?layer_metrics
(regularization_losses
 ?layer_regularization_losses
)	variables
?non_trainable_variables
?layers
*trainable_variables
 
 
 
?
?metrics
?layer_metrics
,regularization_losses
 ?layer_regularization_losses
-	variables
?non_trainable_variables
?layers
.trainable_variables
[Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_24/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

00
11

00
11
?
?metrics
?layer_metrics
2regularization_losses
 ?layer_regularization_losses
3	variables
?non_trainable_variables
?layers
4trainable_variables
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
?
?metrics
?layer_metrics
8regularization_losses
 ?layer_regularization_losses
9	variables
?non_trainable_variables
?layers
:trainable_variables
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
?
?metrics
?layer_metrics
>regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
@trainable_variables
^\
VARIABLE_VALUEmain_output/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEmain_output/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

B0
C1
?
?metrics
?layer_metrics
Dregularization_losses
 ?layer_regularization_losses
E	variables
?non_trainable_variables
?layers
Ftrainable_variables
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
qo
VARIABLE_VALUE5token_and_position_embedding_2/embedding_4/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE5token_and_position_embedding_2/embedding_5/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE?transformer_block_2/multi_head_self_attention_2/dense_18/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE=transformer_block_2/multi_head_self_attention_2/dense_18/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE?transformer_block_2/multi_head_self_attention_2/dense_19/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE=transformer_block_2/multi_head_self_attention_2/dense_19/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE?transformer_block_2/multi_head_self_attention_2/dense_20/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE=transformer_block_2/multi_head_self_attention_2/dense_20/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE?transformer_block_2/multi_head_self_attention_2/dense_21/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE=transformer_block_2/multi_head_self_attention_2/dense_21/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_22/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_22/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdense_23/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_23/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_2/layer_normalization_4/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_2/layer_normalization_4/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE/transformer_block_2/layer_normalization_5/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.transformer_block_2/layer_normalization_5/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE
(
?0
?1
?2
?3
?4
 
 
 
N
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
 

M0

M0
?
?metrics
?layer_metrics
dregularization_losses
 ?layer_regularization_losses
e	variables
?non_trainable_variables
?layers
ftrainable_variables
 

N0

N0
?
?metrics
?layer_metrics
hregularization_losses
 ?layer_regularization_losses
i	variables
?non_trainable_variables
?layers
jtrainable_variables
 
 
 
 

0
1
l

Okernel
Pbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
l

Qkernel
Rbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
l

Skernel
Tbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
l

Ukernel
Vbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 
8
O0
P1
Q2
R3
S4
T5
U6
V7
8
O0
P1
Q2
R3
S4
T5
U6
V7
?
?metrics
?layer_metrics
uregularization_losses
 ?layer_regularization_losses
v	variables
?non_trainable_variables
?layers
wtrainable_variables
l

Wkernel
Xbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
l

Ykernel
Zbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
 

W0
X1
Y2
Z3

W0
X1
Y2
Z3
?
?metrics
?layer_metrics
{regularization_losses
 ?layer_regularization_losses
|	variables
?non_trainable_variables
?layers
}trainable_variables
 
 

[0
\1

[0
\1
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
 
 

]0
^1

]0
^1
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
 
 
 
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
 
 
 
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
 
 
 
 
*
0
1
2
3
4
5
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

?count
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
O0
P1

O0
P1
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
 

Q0
R1

Q0
R1
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
 

S0
T1

S0
T1
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
 

U0
V1

U0
V1
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
 
 
 
 

q0
r1
s2
t3
 

W0
X1

W0
X1
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
 

Y0
Z1

Y0
Z1
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
 
 
 
 

y0
z1
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
?1
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
~|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/main_output/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/main_output/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_4/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_5/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_22/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_22/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_23/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_23/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_4/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_4/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_5/gamma/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_5/beta/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/aux_output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/aux_output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_25/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_25/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_26/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_26/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/main_output/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/main_output/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_4/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE<Adam/token_and_position_embedding_2/embedding_5/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_22/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_22/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/dense_23/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_23/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_4/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_4/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_2/layer_normalization_5/gamma/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/transformer_block_2/layer_normalization_5/beta/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_aux_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_3Placeholder*'
_output_shapes
:?????????(*
dtype0*
shape:?????????(
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_aux_inputserving_default_input_35token_and_position_embedding_2/embedding_5/embeddings5token_and_position_embedding_2/embedding_4/embeddings?transformer_block_2/multi_head_self_attention_2/dense_18/kernel=transformer_block_2/multi_head_self_attention_2/dense_18/bias?transformer_block_2/multi_head_self_attention_2/dense_19/kernel=transformer_block_2/multi_head_self_attention_2/dense_19/bias?transformer_block_2/multi_head_self_attention_2/dense_20/kernel=transformer_block_2/multi_head_self_attention_2/dense_20/bias?transformer_block_2/multi_head_self_attention_2/dense_21/kernel=transformer_block_2/multi_head_self_attention_2/dense_21/bias/transformer_block_2/layer_normalization_4/gamma.transformer_block_2/layer_normalization_4/betadense_22/kerneldense_22/biasdense_23/kerneldense_23/bias/transformer_block_2/layer_normalization_5/gamma.transformer_block_2/layer_normalization_5/betaaux_output/kernelaux_output/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasmain_output/kernelmain_output/bias*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1352099
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?/
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%aux_output/kernel/Read/ReadVariableOp#aux_output/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp&main_output/kernel/Read/ReadVariableOp$main_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_4/embeddings/Read/ReadVariableOpItoken_and_position_embedding_2/embedding_5/embeddings/Read/ReadVariableOpStransformer_block_2/multi_head_self_attention_2/dense_18/kernel/Read/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_18/bias/Read/ReadVariableOpStransformer_block_2/multi_head_self_attention_2/dense_19/kernel/Read/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_19/bias/Read/ReadVariableOpStransformer_block_2/multi_head_self_attention_2/dense_20/kernel/Read/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_20/bias/Read/ReadVariableOpStransformer_block_2/multi_head_self_attention_2/dense_21/kernel/Read/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_21/bias/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpCtransformer_block_2/layer_normalization_4/gamma/Read/ReadVariableOpBtransformer_block_2/layer_normalization_4/beta/Read/ReadVariableOpCtransformer_block_2/layer_normalization_5/gamma/Read/ReadVariableOpBtransformer_block_2/layer_normalization_5/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOp,Adam/aux_output/kernel/m/Read/ReadVariableOp*Adam/aux_output/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp*Adam/dense_25/kernel/m/Read/ReadVariableOp(Adam/dense_25/bias/m/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp-Adam/main_output/kernel/m/Read/ReadVariableOp+Adam/main_output/bias/m/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_4/embeddings/m/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_5/embeddings/m/Read/ReadVariableOpZAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/m/Read/ReadVariableOpXAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/m/Read/ReadVariableOpZAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/m/Read/ReadVariableOpXAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/m/Read/ReadVariableOpZAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/m/Read/ReadVariableOpXAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/m/Read/ReadVariableOpZAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/m/Read/ReadVariableOpXAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_4/gamma/m/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_4/beta/m/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_5/gamma/m/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_5/beta/m/Read/ReadVariableOp,Adam/aux_output/kernel/v/Read/ReadVariableOp*Adam/aux_output/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOp*Adam/dense_25/kernel/v/Read/ReadVariableOp(Adam/dense_25/bias/v/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp-Adam/main_output/kernel/v/Read/ReadVariableOp+Adam/main_output/bias/v/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_4/embeddings/v/Read/ReadVariableOpPAdam/token_and_position_embedding_2/embedding_5/embeddings/v/Read/ReadVariableOpZAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/v/Read/ReadVariableOpXAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/v/Read/ReadVariableOpZAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/v/Read/ReadVariableOpXAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/v/Read/ReadVariableOpZAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/v/Read/ReadVariableOpXAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/v/Read/ReadVariableOpZAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/v/Read/ReadVariableOpXAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_4/gamma/v/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_4/beta/v/Read/ReadVariableOpJAdam/transformer_block_2/layer_normalization_5/gamma/v/Read/ReadVariableOpIAdam/transformer_block_2/layer_normalization_5/beta/v/Read/ReadVariableOpConst*p
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
 __inference__traced_save_1354136
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameaux_output/kernelaux_output/biasdense_24/kerneldense_24/biasdense_25/kerneldense_25/biasdense_26/kerneldense_26/biasmain_output/kernelmain_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate5token_and_position_embedding_2/embedding_4/embeddings5token_and_position_embedding_2/embedding_5/embeddings?transformer_block_2/multi_head_self_attention_2/dense_18/kernel=transformer_block_2/multi_head_self_attention_2/dense_18/bias?transformer_block_2/multi_head_self_attention_2/dense_19/kernel=transformer_block_2/multi_head_self_attention_2/dense_19/bias?transformer_block_2/multi_head_self_attention_2/dense_20/kernel=transformer_block_2/multi_head_self_attention_2/dense_20/bias?transformer_block_2/multi_head_self_attention_2/dense_21/kernel=transformer_block_2/multi_head_self_attention_2/dense_21/biasdense_22/kerneldense_22/biasdense_23/kerneldense_23/bias/transformer_block_2/layer_normalization_4/gamma.transformer_block_2/layer_normalization_4/beta/transformer_block_2/layer_normalization_5/gamma.transformer_block_2/layer_normalization_5/betatotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4Adam/aux_output/kernel/mAdam/aux_output/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/dense_25/kernel/mAdam/dense_25/bias/mAdam/dense_26/kernel/mAdam/dense_26/bias/mAdam/main_output/kernel/mAdam/main_output/bias/m<Adam/token_and_position_embedding_2/embedding_4/embeddings/m<Adam/token_and_position_embedding_2/embedding_5/embeddings/mFAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/mDAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/mFAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/mDAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/mFAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/mDAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/mFAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/mDAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/m6Adam/transformer_block_2/layer_normalization_4/gamma/m5Adam/transformer_block_2/layer_normalization_4/beta/m6Adam/transformer_block_2/layer_normalization_5/gamma/m5Adam/transformer_block_2/layer_normalization_5/beta/mAdam/aux_output/kernel/vAdam/aux_output/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/vAdam/dense_25/kernel/vAdam/dense_25/bias/vAdam/dense_26/kernel/vAdam/dense_26/bias/vAdam/main_output/kernel/vAdam/main_output/bias/v<Adam/token_and_position_embedding_2/embedding_4/embeddings/v<Adam/token_and_position_embedding_2/embedding_5/embeddings/vFAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/vDAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/vFAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/vDAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/vFAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/vDAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/vFAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/vDAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/v6Adam/transformer_block_2/layer_normalization_4/gamma/v5Adam/transformer_block_2/layer_normalization_4/beta/v6Adam/transformer_block_2/layer_normalization_5/gamma/v5Adam/transformer_block_2/layer_normalization_5/beta/v*o
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
#__inference__traced_restore_1354443ƒ,
?
?
G__inference_aux_output_layer_call_and_return_conditional_losses_1353502

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
??
? 
D__inference_model_2_layer_call_and_return_conditional_losses_1352851
inputs_0
inputs_1U
Ctoken_and_position_embedding_2_embedding_5_embedding_lookup_1352544:( U
Ctoken_and_position_embedding_2_embedding_4_embedding_lookup_1352550: l
Ztransformer_block_2_multi_head_self_attention_2_dense_18_tensordot_readvariableop_resource:  f
Xtransformer_block_2_multi_head_self_attention_2_dense_18_biasadd_readvariableop_resource: l
Ztransformer_block_2_multi_head_self_attention_2_dense_19_tensordot_readvariableop_resource:  f
Xtransformer_block_2_multi_head_self_attention_2_dense_19_biasadd_readvariableop_resource: l
Ztransformer_block_2_multi_head_self_attention_2_dense_20_tensordot_readvariableop_resource:  f
Xtransformer_block_2_multi_head_self_attention_2_dense_20_biasadd_readvariableop_resource: l
Ztransformer_block_2_multi_head_self_attention_2_dense_21_tensordot_readvariableop_resource:  f
Xtransformer_block_2_multi_head_self_attention_2_dense_21_biasadd_readvariableop_resource: ]
Otransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource: Y
Ktransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource: ]
Ktransformer_block_2_sequential_2_dense_22_tensordot_readvariableop_resource:  W
Itransformer_block_2_sequential_2_dense_22_biasadd_readvariableop_resource: ]
Ktransformer_block_2_sequential_2_dense_23_tensordot_readvariableop_resource:  W
Itransformer_block_2_sequential_2_dense_23_biasadd_readvariableop_resource: ]
Otransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource: Y
Ktransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource: ;
)aux_output_matmul_readvariableop_resource: 8
*aux_output_biasadd_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource:@6
(dense_24_biasadd_readvariableop_resource:@9
'dense_25_matmul_readvariableop_resource:@@6
(dense_25_biasadd_readvariableop_resource:@9
'dense_26_matmul_readvariableop_resource:@@6
(dense_26_biasadd_readvariableop_resource:@<
*main_output_matmul_readvariableop_resource:@9
+main_output_biasadd_readvariableop_resource:
identity

identity_1??!aux_output/BiasAdd/ReadVariableOp? aux_output/MatMul/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?"main_output/BiasAdd/ReadVariableOp?!main_output/MatMul/ReadVariableOp?;token_and_position_embedding_2/embedding_4/embedding_lookup?;token_and_position_embedding_2/embedding_5/embedding_lookup?Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp?Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp?Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp?Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp?Otransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?Otransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?Otransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?Otransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?@transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp?Btransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp?@transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp?Btransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp?
$token_and_position_embedding_2/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_2/Shape?
2token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2token_and_position_embedding_2/strided_slice/stack?
4token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_2/strided_slice/stack_1?
4token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_2/strided_slice/stack_2?
,token_and_position_embedding_2/strided_sliceStridedSlice-token_and_position_embedding_2/Shape:output:0;token_and_position_embedding_2/strided_slice/stack:output:0=token_and_position_embedding_2/strided_slice/stack_1:output:0=token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_2/strided_slice?
*token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_2/range/start?
*token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_2/range/delta?
$token_and_position_embedding_2/rangeRange3token_and_position_embedding_2/range/start:output:05token_and_position_embedding_2/strided_slice:output:03token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:?????????2&
$token_and_position_embedding_2/range?
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherCtoken_and_position_embedding_2_embedding_5_embedding_lookup_1352544-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/1352544*'
_output_shapes
:????????? *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookup?
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/1352544*'
_output_shapes
:????????? 2F
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity?
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2H
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1?
/token_and_position_embedding_2/embedding_4/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????(21
/token_and_position_embedding_2/embedding_4/Cast?
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherCtoken_and_position_embedding_2_embedding_4_embedding_lookup_13525503token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/1352550*+
_output_shapes
:?????????( *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookup?
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/1352550*+
_output_shapes
:?????????( 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity?
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1?
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2$
"token_and_position_embedding_2/add?
5transformer_block_2/multi_head_self_attention_2/ShapeShape&token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:27
5transformer_block_2/multi_head_self_attention_2/Shape?
Ctransformer_block_2/multi_head_self_attention_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_2/multi_head_self_attention_2/strided_slice/stack?
Etransformer_block_2/multi_head_self_attention_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Etransformer_block_2/multi_head_self_attention_2/strided_slice/stack_1?
Etransformer_block_2/multi_head_self_attention_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Etransformer_block_2/multi_head_self_attention_2/strided_slice/stack_2?
=transformer_block_2/multi_head_self_attention_2/strided_sliceStridedSlice>transformer_block_2/multi_head_self_attention_2/Shape:output:0Ltransformer_block_2/multi_head_self_attention_2/strided_slice/stack:output:0Ntransformer_block_2/multi_head_self_attention_2/strided_slice/stack_1:output:0Ntransformer_block_2/multi_head_self_attention_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=transformer_block_2/multi_head_self_attention_2/strided_slice?
Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_2_multi_head_self_attention_2_dense_18_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axes?
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/free?
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ShapeShape&token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Shape?
Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/free:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2?
Rtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis?
Mtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axes:output:0[transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1?
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const?
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ProdProdTtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod?
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_1?
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod_1ProdVtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1:output:0Stransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod_1?
Ntransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat/axis?
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concatConcatV2Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/free:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axes:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat?
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/stackPackPtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod:output:0Rtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/stack?
Ltransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/transpose	Transpose&token_and_position_embedding_2/add:z:0Rtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/transpose?
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReshapeReshapePtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/transpose:y:0Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Reshape?
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/MatMulMatMulStransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Reshape:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/MatMul?
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_2?
Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1ConcatV2Ttransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0Stransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_2:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1?
Btransformer_block_2/multi_head_self_attention_2/dense_18/TensordotReshapeStransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/MatMul:product:0Ttransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot?
Otransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_2_multi_head_self_attention_2_dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?
@transformer_block_2/multi_head_self_attention_2/dense_18/BiasAddBiasAddKtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd?
Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_2_multi_head_self_attention_2_dense_19_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axes?
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/free?
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ShapeShape&token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Shape?
Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/free:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2?
Rtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis?
Mtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axes:output:0[transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1?
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const?
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ProdProdTtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod?
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_1?
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod_1ProdVtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1:output:0Stransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod_1?
Ntransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat/axis?
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concatConcatV2Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/free:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axes:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat?
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/stackPackPtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod:output:0Rtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/stack?
Ltransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/transpose	Transpose&token_and_position_embedding_2/add:z:0Rtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/transpose?
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReshapeReshapePtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/transpose:y:0Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Reshape?
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/MatMulMatMulStransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Reshape:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/MatMul?
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_2?
Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1ConcatV2Ttransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0Stransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_2:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1?
Btransformer_block_2/multi_head_self_attention_2/dense_19/TensordotReshapeStransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/MatMul:product:0Ttransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot?
Otransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_2_multi_head_self_attention_2_dense_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?
@transformer_block_2/multi_head_self_attention_2/dense_19/BiasAddBiasAddKtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd?
Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_2_multi_head_self_attention_2_dense_20_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axes?
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/free?
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ShapeShape&token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Shape?
Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/free:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2?
Rtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis?
Mtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axes:output:0[transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1?
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const?
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ProdProdTtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod?
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_1?
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod_1ProdVtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1:output:0Stransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod_1?
Ntransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat/axis?
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concatConcatV2Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/free:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axes:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat?
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/stackPackPtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod:output:0Rtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/stack?
Ltransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/transpose	Transpose&token_and_position_embedding_2/add:z:0Rtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/transpose?
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReshapeReshapePtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/transpose:y:0Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Reshape?
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/MatMulMatMulStransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Reshape:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/MatMul?
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_2?
Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1ConcatV2Ttransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0Stransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_2:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1?
Btransformer_block_2/multi_head_self_attention_2/dense_20/TensordotReshapeStransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/MatMul:product:0Ttransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot?
Otransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_2_multi_head_self_attention_2_dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?
@transformer_block_2/multi_head_self_attention_2/dense_20/BiasAddBiasAddKtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd?
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2A
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/1?
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2A
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/2?
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2A
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/3?
=transformer_block_2/multi_head_self_attention_2/Reshape/shapePackFtransformer_block_2/multi_head_self_attention_2/strided_slice:output:0Htransformer_block_2/multi_head_self_attention_2/Reshape/shape/1:output:0Htransformer_block_2/multi_head_self_attention_2/Reshape/shape/2:output:0Htransformer_block_2/multi_head_self_attention_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_2/multi_head_self_attention_2/Reshape/shape?
7transformer_block_2/multi_head_self_attention_2/ReshapeReshapeItransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd:output:0Ftransformer_block_2/multi_head_self_attention_2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????29
7transformer_block_2/multi_head_self_attention_2/Reshape?
>transformer_block_2/multi_head_self_attention_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2@
>transformer_block_2/multi_head_self_attention_2/transpose/perm?
9transformer_block_2/multi_head_self_attention_2/transpose	Transpose@transformer_block_2/multi_head_self_attention_2/Reshape:output:0Gtransformer_block_2/multi_head_self_attention_2/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_2/multi_head_self_attention_2/transpose?
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/1?
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/2?
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/3?
?transformer_block_2/multi_head_self_attention_2/Reshape_1/shapePackFtransformer_block_2/multi_head_self_attention_2/strided_slice:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/1:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/2:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_2/multi_head_self_attention_2/Reshape_1/shape?
9transformer_block_2/multi_head_self_attention_2/Reshape_1ReshapeItransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd:output:0Htransformer_block_2/multi_head_self_attention_2/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_2/multi_head_self_attention_2/Reshape_1?
@transformer_block_2/multi_head_self_attention_2/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_2/multi_head_self_attention_2/transpose_1/perm?
;transformer_block_2/multi_head_self_attention_2/transpose_1	TransposeBtransformer_block_2/multi_head_self_attention_2/Reshape_1:output:0Itransformer_block_2/multi_head_self_attention_2/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_2/multi_head_self_attention_2/transpose_1?
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/1?
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/2?
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/3?
?transformer_block_2/multi_head_self_attention_2/Reshape_2/shapePackFtransformer_block_2/multi_head_self_attention_2/strided_slice:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/1:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/2:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_2/multi_head_self_attention_2/Reshape_2/shape?
9transformer_block_2/multi_head_self_attention_2/Reshape_2ReshapeItransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd:output:0Htransformer_block_2/multi_head_self_attention_2/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_2/multi_head_self_attention_2/Reshape_2?
@transformer_block_2/multi_head_self_attention_2/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_2/multi_head_self_attention_2/transpose_2/perm?
;transformer_block_2/multi_head_self_attention_2/transpose_2	TransposeBtransformer_block_2/multi_head_self_attention_2/Reshape_2:output:0Itransformer_block_2/multi_head_self_attention_2/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_2/multi_head_self_attention_2/transpose_2?
6transformer_block_2/multi_head_self_attention_2/MatMulBatchMatMulV2=transformer_block_2/multi_head_self_attention_2/transpose:y:0?transformer_block_2/multi_head_self_attention_2/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(28
6transformer_block_2/multi_head_self_attention_2/MatMul?
7transformer_block_2/multi_head_self_attention_2/Shape_1Shape?transformer_block_2/multi_head_self_attention_2/transpose_1:y:0*
T0*
_output_shapes
:29
7transformer_block_2/multi_head_self_attention_2/Shape_1?
Etransformer_block_2/multi_head_self_attention_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2G
Etransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack?
Gtransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gtransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_1?
Gtransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_2?
?transformer_block_2/multi_head_self_attention_2/strided_slice_1StridedSlice@transformer_block_2/multi_head_self_attention_2/Shape_1:output:0Ntransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack:output:0Ptransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_1:output:0Ptransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?transformer_block_2/multi_head_self_attention_2/strided_slice_1?
4transformer_block_2/multi_head_self_attention_2/CastCastHtransformer_block_2/multi_head_self_attention_2/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 26
4transformer_block_2/multi_head_self_attention_2/Cast?
4transformer_block_2/multi_head_self_attention_2/SqrtSqrt8transformer_block_2/multi_head_self_attention_2/Cast:y:0*
T0*
_output_shapes
: 26
4transformer_block_2/multi_head_self_attention_2/Sqrt?
7transformer_block_2/multi_head_self_attention_2/truedivRealDiv?transformer_block_2/multi_head_self_attention_2/MatMul:output:08transformer_block_2/multi_head_self_attention_2/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????29
7transformer_block_2/multi_head_self_attention_2/truediv?
7transformer_block_2/multi_head_self_attention_2/SoftmaxSoftmax;transformer_block_2/multi_head_self_attention_2/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????29
7transformer_block_2/multi_head_self_attention_2/Softmax?
8transformer_block_2/multi_head_self_attention_2/MatMul_1BatchMatMulV2Atransformer_block_2/multi_head_self_attention_2/Softmax:softmax:0?transformer_block_2/multi_head_self_attention_2/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2:
8transformer_block_2/multi_head_self_attention_2/MatMul_1?
@transformer_block_2/multi_head_self_attention_2/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_2/multi_head_self_attention_2/transpose_3/perm?
;transformer_block_2/multi_head_self_attention_2/transpose_3	TransposeAtransformer_block_2/multi_head_self_attention_2/MatMul_1:output:0Itransformer_block_2/multi_head_self_attention_2/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_2/multi_head_self_attention_2/transpose_3?
Atransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/1?
Atransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/2?
?transformer_block_2/multi_head_self_attention_2/Reshape_3/shapePackFtransformer_block_2/multi_head_self_attention_2/strided_slice:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/1:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_2/multi_head_self_attention_2/Reshape_3/shape?
9transformer_block_2/multi_head_self_attention_2/Reshape_3Reshape?transformer_block_2/multi_head_self_attention_2/transpose_3:y:0Htransformer_block_2/multi_head_self_attention_2/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2;
9transformer_block_2/multi_head_self_attention_2/Reshape_3?
Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_2_multi_head_self_attention_2_dense_21_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axes?
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/free?
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ShapeShapeBtransformer_block_2/multi_head_self_attention_2/Reshape_3:output:0*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Shape?
Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/free:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2?
Rtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis?
Mtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axes:output:0[transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1?
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const?
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ProdProdTtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod?
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_1?
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod_1ProdVtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1:output:0Stransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod_1?
Ntransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat/axis?
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concatConcatV2Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/free:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axes:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat?
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/stackPackPtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod:output:0Rtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/stack?
Ltransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/transpose	TransposeBtransformer_block_2/multi_head_self_attention_2/Reshape_3:output:0Rtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2N
Ltransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/transpose?
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReshapeReshapePtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/transpose:y:0Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Reshape?
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/MatMulMatMulStransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Reshape:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/MatMul?
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_2?
Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1ConcatV2Ttransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0Stransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_2:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1?
Btransformer_block_2/multi_head_self_attention_2/dense_21/TensordotReshapeStransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/MatMul:product:0Ttransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 2D
Btransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot?
Otransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_2_multi_head_self_attention_2_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?
@transformer_block_2/multi_head_self_attention_2/dense_21/BiasAddBiasAddKtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2B
@transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd?
+transformer_block_2/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2-
+transformer_block_2/dropout_4/dropout/Const?
)transformer_block_2/dropout_4/dropout/MulMulItransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd:output:04transformer_block_2/dropout_4/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2+
)transformer_block_2/dropout_4/dropout/Mul?
+transformer_block_2/dropout_4/dropout/ShapeShapeItransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd:output:0*
T0*
_output_shapes
:2-
+transformer_block_2/dropout_4/dropout/Shape?
Btransformer_block_2/dropout_4/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_2/dropout_4/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype02D
Btransformer_block_2/dropout_4/dropout/random_uniform/RandomUniform?
4transformer_block_2/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=26
4transformer_block_2/dropout_4/dropout/GreaterEqual/y?
2transformer_block_2/dropout_4/dropout/GreaterEqualGreaterEqualKtransformer_block_2/dropout_4/dropout/random_uniform/RandomUniform:output:0=transformer_block_2/dropout_4/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 24
2transformer_block_2/dropout_4/dropout/GreaterEqual?
*transformer_block_2/dropout_4/dropout/CastCast6transformer_block_2/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2,
*transformer_block_2/dropout_4/dropout/Cast?
+transformer_block_2/dropout_4/dropout/Mul_1Mul-transformer_block_2/dropout_4/dropout/Mul:z:0.transformer_block_2/dropout_4/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2-
+transformer_block_2/dropout_4/dropout/Mul_1?
transformer_block_2/addAddV2&token_and_position_embedding_2/add:z:0/transformer_block_2/dropout_4/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_2/add?
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indices?
6transformer_block_2/layer_normalization_4/moments/meanMeantransformer_block_2/add:z:0Qtransformer_block_2/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(28
6transformer_block_2/layer_normalization_4/moments/mean?
>transformer_block_2/layer_normalization_4/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2@
>transformer_block_2/layer_normalization_4/moments/StopGradient?
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add:z:0Gtransformer_block_2/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2E
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifference?
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indices?
:transformer_block_2/layer_normalization_4/moments/varianceMeanGtransformer_block_2/layer_normalization_4/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2<
:transformer_block_2/layer_normalization_4/moments/variance?
9transformer_block_2/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_2/layer_normalization_4/batchnorm/add/y?
7transformer_block_2/layer_normalization_4/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_4/moments/variance:output:0Btransformer_block_2/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(29
7transformer_block_2/layer_normalization_4/batchnorm/add?
9transformer_block_2/layer_normalization_4/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2;
9transformer_block_2/layer_normalization_4/batchnorm/Rsqrt?
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp?
7transformer_block_2/layer_normalization_4/batchnorm/mulMul=transformer_block_2/layer_normalization_4/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 29
7transformer_block_2/layer_normalization_4/batchnorm/mul?
9transformer_block_2/layer_normalization_4/batchnorm/mul_1Multransformer_block_2/add:z:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_1?
9transformer_block_2/layer_normalization_4/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_4/moments/mean:output:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_2?
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp?
7transformer_block_2/layer_normalization_4/batchnorm/subSubJtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 29
7transformer_block_2/layer_normalization_4/batchnorm/sub?
9transformer_block_2/layer_normalization_4/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_4/batchnorm/add_1?
Btransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_2_sequential_2_dense_22_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02D
Btransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp?
8transformer_block_2/sequential_2/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_2/sequential_2/dense_22/Tensordot/axes?
8transformer_block_2/sequential_2/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_2/sequential_2/dense_22/Tensordot/free?
9transformer_block_2/sequential_2/dense_22/Tensordot/ShapeShape=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_22/Tensordot/Shape?
Atransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2/axis?
<transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2GatherV2Btransformer_block_2/sequential_2/dense_22/Tensordot/Shape:output:0Atransformer_block_2/sequential_2/dense_22/Tensordot/free:output:0Jtransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2?
Ctransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1/axis?
>transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1GatherV2Btransformer_block_2/sequential_2/dense_22/Tensordot/Shape:output:0Atransformer_block_2/sequential_2/dense_22/Tensordot/axes:output:0Ltransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1?
9transformer_block_2/sequential_2/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_2/sequential_2/dense_22/Tensordot/Const?
8transformer_block_2/sequential_2/dense_22/Tensordot/ProdProdEtransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2:output:0Btransformer_block_2/sequential_2/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_2/sequential_2/dense_22/Tensordot/Prod?
;transformer_block_2/sequential_2/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_2/sequential_2/dense_22/Tensordot/Const_1?
:transformer_block_2/sequential_2/dense_22/Tensordot/Prod_1ProdGtransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1:output:0Dtransformer_block_2/sequential_2/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_2/sequential_2/dense_22/Tensordot/Prod_1?
?transformer_block_2/sequential_2/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_2/sequential_2/dense_22/Tensordot/concat/axis?
:transformer_block_2/sequential_2/dense_22/Tensordot/concatConcatV2Atransformer_block_2/sequential_2/dense_22/Tensordot/free:output:0Atransformer_block_2/sequential_2/dense_22/Tensordot/axes:output:0Htransformer_block_2/sequential_2/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_2/sequential_2/dense_22/Tensordot/concat?
9transformer_block_2/sequential_2/dense_22/Tensordot/stackPackAtransformer_block_2/sequential_2/dense_22/Tensordot/Prod:output:0Ctransformer_block_2/sequential_2/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_22/Tensordot/stack?
=transformer_block_2/sequential_2/dense_22/Tensordot/transpose	Transpose=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0Ctransformer_block_2/sequential_2/dense_22/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2?
=transformer_block_2/sequential_2/dense_22/Tensordot/transpose?
;transformer_block_2/sequential_2/dense_22/Tensordot/ReshapeReshapeAtransformer_block_2/sequential_2/dense_22/Tensordot/transpose:y:0Btransformer_block_2/sequential_2/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_2/sequential_2/dense_22/Tensordot/Reshape?
:transformer_block_2/sequential_2/dense_22/Tensordot/MatMulMatMulDtransformer_block_2/sequential_2/dense_22/Tensordot/Reshape:output:0Jtransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_2/sequential_2/dense_22/Tensordot/MatMul?
;transformer_block_2/sequential_2/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_2/sequential_2/dense_22/Tensordot/Const_2?
Atransformer_block_2/sequential_2/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_2/sequential_2/dense_22/Tensordot/concat_1/axis?
<transformer_block_2/sequential_2/dense_22/Tensordot/concat_1ConcatV2Etransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2:output:0Dtransformer_block_2/sequential_2/dense_22/Tensordot/Const_2:output:0Jtransformer_block_2/sequential_2/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_2/sequential_2/dense_22/Tensordot/concat_1?
3transformer_block_2/sequential_2/dense_22/TensordotReshapeDtransformer_block_2/sequential_2/dense_22/Tensordot/MatMul:product:0Etransformer_block_2/sequential_2/dense_22/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 25
3transformer_block_2/sequential_2/dense_22/Tensordot?
@transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_2_sequential_2_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp?
1transformer_block_2/sequential_2/dense_22/BiasAddBiasAdd<transformer_block_2/sequential_2/dense_22/Tensordot:output:0Htransformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 23
1transformer_block_2/sequential_2/dense_22/BiasAdd?
.transformer_block_2/sequential_2/dense_22/ReluRelu:transformer_block_2/sequential_2/dense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 20
.transformer_block_2/sequential_2/dense_22/Relu?
Btransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_2_sequential_2_dense_23_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02D
Btransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp?
8transformer_block_2/sequential_2/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_2/sequential_2/dense_23/Tensordot/axes?
8transformer_block_2/sequential_2/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_2/sequential_2/dense_23/Tensordot/free?
9transformer_block_2/sequential_2/dense_23/Tensordot/ShapeShape<transformer_block_2/sequential_2/dense_22/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_23/Tensordot/Shape?
Atransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2/axis?
<transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2GatherV2Btransformer_block_2/sequential_2/dense_23/Tensordot/Shape:output:0Atransformer_block_2/sequential_2/dense_23/Tensordot/free:output:0Jtransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2?
Ctransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1/axis?
>transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1GatherV2Btransformer_block_2/sequential_2/dense_23/Tensordot/Shape:output:0Atransformer_block_2/sequential_2/dense_23/Tensordot/axes:output:0Ltransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1?
9transformer_block_2/sequential_2/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_2/sequential_2/dense_23/Tensordot/Const?
8transformer_block_2/sequential_2/dense_23/Tensordot/ProdProdEtransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2:output:0Btransformer_block_2/sequential_2/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_2/sequential_2/dense_23/Tensordot/Prod?
;transformer_block_2/sequential_2/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_2/sequential_2/dense_23/Tensordot/Const_1?
:transformer_block_2/sequential_2/dense_23/Tensordot/Prod_1ProdGtransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1:output:0Dtransformer_block_2/sequential_2/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_2/sequential_2/dense_23/Tensordot/Prod_1?
?transformer_block_2/sequential_2/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_2/sequential_2/dense_23/Tensordot/concat/axis?
:transformer_block_2/sequential_2/dense_23/Tensordot/concatConcatV2Atransformer_block_2/sequential_2/dense_23/Tensordot/free:output:0Atransformer_block_2/sequential_2/dense_23/Tensordot/axes:output:0Htransformer_block_2/sequential_2/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_2/sequential_2/dense_23/Tensordot/concat?
9transformer_block_2/sequential_2/dense_23/Tensordot/stackPackAtransformer_block_2/sequential_2/dense_23/Tensordot/Prod:output:0Ctransformer_block_2/sequential_2/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_23/Tensordot/stack?
=transformer_block_2/sequential_2/dense_23/Tensordot/transpose	Transpose<transformer_block_2/sequential_2/dense_22/Relu:activations:0Ctransformer_block_2/sequential_2/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2?
=transformer_block_2/sequential_2/dense_23/Tensordot/transpose?
;transformer_block_2/sequential_2/dense_23/Tensordot/ReshapeReshapeAtransformer_block_2/sequential_2/dense_23/Tensordot/transpose:y:0Btransformer_block_2/sequential_2/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_2/sequential_2/dense_23/Tensordot/Reshape?
:transformer_block_2/sequential_2/dense_23/Tensordot/MatMulMatMulDtransformer_block_2/sequential_2/dense_23/Tensordot/Reshape:output:0Jtransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_2/sequential_2/dense_23/Tensordot/MatMul?
;transformer_block_2/sequential_2/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_2/sequential_2/dense_23/Tensordot/Const_2?
Atransformer_block_2/sequential_2/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_2/sequential_2/dense_23/Tensordot/concat_1/axis?
<transformer_block_2/sequential_2/dense_23/Tensordot/concat_1ConcatV2Etransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2:output:0Dtransformer_block_2/sequential_2/dense_23/Tensordot/Const_2:output:0Jtransformer_block_2/sequential_2/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_2/sequential_2/dense_23/Tensordot/concat_1?
3transformer_block_2/sequential_2/dense_23/TensordotReshapeDtransformer_block_2/sequential_2/dense_23/Tensordot/MatMul:product:0Etransformer_block_2/sequential_2/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 25
3transformer_block_2/sequential_2/dense_23/Tensordot?
@transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_2_sequential_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp?
1transformer_block_2/sequential_2/dense_23/BiasAddBiasAdd<transformer_block_2/sequential_2/dense_23/Tensordot:output:0Htransformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 23
1transformer_block_2/sequential_2/dense_23/BiasAdd?
+transformer_block_2/dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2-
+transformer_block_2/dropout_5/dropout/Const?
)transformer_block_2/dropout_5/dropout/MulMul:transformer_block_2/sequential_2/dense_23/BiasAdd:output:04transformer_block_2/dropout_5/dropout/Const:output:0*
T0*+
_output_shapes
:?????????( 2+
)transformer_block_2/dropout_5/dropout/Mul?
+transformer_block_2/dropout_5/dropout/ShapeShape:transformer_block_2/sequential_2/dense_23/BiasAdd:output:0*
T0*
_output_shapes
:2-
+transformer_block_2/dropout_5/dropout/Shape?
Btransformer_block_2/dropout_5/dropout/random_uniform/RandomUniformRandomUniform4transformer_block_2/dropout_5/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????( *
dtype02D
Btransformer_block_2/dropout_5/dropout/random_uniform/RandomUniform?
4transformer_block_2/dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=26
4transformer_block_2/dropout_5/dropout/GreaterEqual/y?
2transformer_block_2/dropout_5/dropout/GreaterEqualGreaterEqualKtransformer_block_2/dropout_5/dropout/random_uniform/RandomUniform:output:0=transformer_block_2/dropout_5/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????( 24
2transformer_block_2/dropout_5/dropout/GreaterEqual?
*transformer_block_2/dropout_5/dropout/CastCast6transformer_block_2/dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????( 2,
*transformer_block_2/dropout_5/dropout/Cast?
+transformer_block_2/dropout_5/dropout/Mul_1Mul-transformer_block_2/dropout_5/dropout/Mul:z:0.transformer_block_2/dropout_5/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????( 2-
+transformer_block_2/dropout_5/dropout/Mul_1?
transformer_block_2/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0/transformer_block_2/dropout_5/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_2/add_1?
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indices?
6transformer_block_2/layer_normalization_5/moments/meanMeantransformer_block_2/add_1:z:0Qtransformer_block_2/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(28
6transformer_block_2/layer_normalization_5/moments/mean?
>transformer_block_2/layer_normalization_5/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2@
>transformer_block_2/layer_normalization_5/moments/StopGradient?
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add_1:z:0Gtransformer_block_2/layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2E
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifference?
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indices?
:transformer_block_2/layer_normalization_5/moments/varianceMeanGtransformer_block_2/layer_normalization_5/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2<
:transformer_block_2/layer_normalization_5/moments/variance?
9transformer_block_2/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_2/layer_normalization_5/batchnorm/add/y?
7transformer_block_2/layer_normalization_5/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_5/moments/variance:output:0Btransformer_block_2/layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(29
7transformer_block_2/layer_normalization_5/batchnorm/add?
9transformer_block_2/layer_normalization_5/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2;
9transformer_block_2/layer_normalization_5/batchnorm/Rsqrt?
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp?
7transformer_block_2/layer_normalization_5/batchnorm/mulMul=transformer_block_2/layer_normalization_5/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 29
7transformer_block_2/layer_normalization_5/batchnorm/mul?
9transformer_block_2/layer_normalization_5/batchnorm/mul_1Multransformer_block_2/add_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_1?
9transformer_block_2/layer_normalization_5/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_5/moments/mean:output:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_2?
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp?
7transformer_block_2/layer_normalization_5/batchnorm/subSubJtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 29
7transformer_block_2/layer_normalization_5/batchnorm/sub?
9transformer_block_2/layer_normalization_5/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_5/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_5/batchnorm/add_1?
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices?
global_average_pooling1d_2/MeanMean=transformer_block_2/layer_normalization_5/batchnorm/add_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2!
global_average_pooling1d_2/Mean?
 aux_output/MatMul/ReadVariableOpReadVariableOp)aux_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 aux_output/MatMul/ReadVariableOp?
aux_output/MatMulMatMul(global_average_pooling1d_2/Mean:output:0(aux_output/MatMul/ReadVariableOp:value:0*
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
aux_output/Sigmoidx
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV2aux_output/Sigmoid:y:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_2/concat?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMulconcatenate_2/concat:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_24/BiasAdds
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_24/Relu?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_25/Relu?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_26/MatMul/ReadVariableOp?
dense_26/MatMulMatMuldense_25/Relu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_26/MatMul?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_26/BiasAdd/ReadVariableOp?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_26/Relu?
!main_output/MatMul/ReadVariableOpReadVariableOp*main_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!main_output/MatMul/ReadVariableOp?
main_output/MatMulMatMuldense_26/Relu:activations:0)main_output/MatMul/ReadVariableOp:value:0*
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

Identity_1?
NoOpNoOp"^aux_output/BiasAdd/ReadVariableOp!^aux_output/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp#^main_output/BiasAdd/ReadVariableOp"^main_output/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupC^transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpC^transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpP^transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpR^transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpP^transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpR^transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpP^transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpR^transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpP^transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpR^transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpA^transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOpC^transformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOpA^transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOpC^transformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!aux_output/BiasAdd/ReadVariableOp!aux_output/BiasAdd/ReadVariableOp2D
 aux_output/MatMul/ReadVariableOp aux_output/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2H
"main_output/BiasAdd/ReadVariableOp"main_output/BiasAdd/ReadVariableOp2F
!main_output/MatMul/ReadVariableOp!main_output/MatMul/ReadVariableOp2z
;token_and_position_embedding_2/embedding_4/embedding_lookup;token_and_position_embedding_2/embedding_4/embedding_lookup2z
;token_and_position_embedding_2/embedding_5/embedding_lookup;token_and_position_embedding_2/embedding_5/embedding_lookup2?
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp2?
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp2?
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp2?
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp2?
Otransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpOtransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp2?
Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp2?
Otransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpOtransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp2?
Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp2?
Otransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpOtransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp2?
Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp2?
Otransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpOtransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp2?
Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp2?
@transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp@transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp2?
Btransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOpBtransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp2?
@transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp@transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp2?
Btransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOpBtransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp:Q M
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
inputs/1
?
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1350600

inputs"
dense_22_1350558:  
dense_22_1350560: "
dense_23_1350594:  
dense_23_1350596: 
identity?? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCallinputsdense_22_1350558dense_22_1350560*
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
GPU 2J 8? *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_13505572"
 dense_22/StatefulPartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1350594dense_23_1350596*
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
GPU 2J 8? *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_13505932"
 dense_23/StatefulPartitionedCall?
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
)__inference_model_2_layer_call_fn_1351885
input_3
	aux_input
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

unknown_19:@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3	aux_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_13517602
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
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????(
!
_user_specified_name	input_3:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input
? 
?
E__inference_dense_23_layer_call_and_return_conditional_losses_1350593

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
?
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1350660

inputs"
dense_22_1350649:  
dense_22_1350651: "
dense_23_1350654:  
dense_23_1350656: 
identity?? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCallinputsdense_22_1350649dense_22_1350651*
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
GPU 2J 8? *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_13505572"
 dense_22/StatefulPartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1350654dense_23_1350656*
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
GPU 2J 8? *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_13505932"
 dense_23/StatefulPartitionedCall?
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
E__inference_dense_26_layer_call_and_return_conditional_losses_1353575

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
H__inference_main_output_layer_call_and_return_conditional_losses_1353595

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
?
?
G__inference_aux_output_layer_call_and_return_conditional_losses_1351071

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
?
?
.__inference_sequential_2_layer_call_fn_1350611
dense_22_input
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_22_inputunknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_13506002
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
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:?????????( 
(
_user_specified_namedense_22_input
?
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1350722

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
?
?
*__inference_dense_24_layer_call_fn_1353524

inputs
unknown:@
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
GPU 2J 8? *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_13510972
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
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_23_layer_call_fn_1353784

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
GPU 2J 8? *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_13505932
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
?
?
%__inference_signature_wrapper_1352099
	aux_input
input_3
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

unknown_19:@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3	aux_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_13505192
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
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????:?????????(: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	aux_input:PL
'
_output_shapes
:?????????(
!
_user_specified_name	input_3
?
?
E__inference_dense_26_layer_call_and_return_conditional_losses_1351131

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
?<
?
D__inference_model_2_layer_call_and_return_conditional_losses_1352027
input_3
	aux_input8
&token_and_position_embedding_2_1351960:( 8
&token_and_position_embedding_2_1351962: -
transformer_block_2_1351965:  )
transformer_block_2_1351967: -
transformer_block_2_1351969:  )
transformer_block_2_1351971: -
transformer_block_2_1351973:  )
transformer_block_2_1351975: -
transformer_block_2_1351977:  )
transformer_block_2_1351979: )
transformer_block_2_1351981: )
transformer_block_2_1351983: -
transformer_block_2_1351985:  )
transformer_block_2_1351987: -
transformer_block_2_1351989:  )
transformer_block_2_1351991: )
transformer_block_2_1351993: )
transformer_block_2_1351995: $
aux_output_1351999:  
aux_output_1352001:"
dense_24_1352005:@
dense_24_1352007:@"
dense_25_1352010:@@
dense_25_1352012:@"
dense_26_1352015:@@
dense_26_1352017:@%
main_output_1352020:@!
main_output_1352022:
identity

identity_1??"aux_output/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?6token_and_position_embedding_2/StatefulPartitionedCall?+transformer_block_2/StatefulPartitionedCall?
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_3&token_and_position_embedding_2_1351960&token_and_position_embedding_2_1351962*
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
GPU 2J 8? *d
f_R]
[__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_135076928
6token_and_position_embedding_2/StatefulPartitionedCall?
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_1351965transformer_block_2_1351967transformer_block_2_1351969transformer_block_2_1351971transformer_block_2_1351973transformer_block_2_1351975transformer_block_2_1351977transformer_block_2_1351979transformer_block_2_1351981transformer_block_2_1351983transformer_block_2_1351985transformer_block_2_1351987transformer_block_2_1351989transformer_block_2_1351991transformer_block_2_1351993transformer_block_2_1351995*
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
GPU 2J 8? *Y
fTRR
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_13515772-
+transformer_block_2/StatefulPartitionedCall?
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13510582,
*global_average_pooling1d_2/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0aux_output_1351999aux_output_1352001*
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
G__inference_aux_output_layer_call_and_return_conditional_losses_13510712$
"aux_output/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0	aux_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_13510842
concatenate_2/PartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_24_1352005dense_24_1352007*
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
GPU 2J 8? *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_13510972"
 dense_24/StatefulPartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_1352010dense_25_1352012*
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
GPU 2J 8? *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_13511142"
 dense_25/StatefulPartitionedCall?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_1352015dense_26_1352017*
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
GPU 2J 8? *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_13511312"
 dense_26/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0main_output_1352020main_output_1352022*
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
H__inference_main_output_layer_call_and_return_conditional_losses_13511482%
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
NoOpNoOp#^aux_output/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall$^main_output/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????(
!
_user_specified_name	input_3:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input
?<
?
D__inference_model_2_layer_call_and_return_conditional_losses_1351956
input_3
	aux_input8
&token_and_position_embedding_2_1351889:( 8
&token_and_position_embedding_2_1351891: -
transformer_block_2_1351894:  )
transformer_block_2_1351896: -
transformer_block_2_1351898:  )
transformer_block_2_1351900: -
transformer_block_2_1351902:  )
transformer_block_2_1351904: -
transformer_block_2_1351906:  )
transformer_block_2_1351908: )
transformer_block_2_1351910: )
transformer_block_2_1351912: -
transformer_block_2_1351914:  )
transformer_block_2_1351916: -
transformer_block_2_1351918:  )
transformer_block_2_1351920: )
transformer_block_2_1351922: )
transformer_block_2_1351924: $
aux_output_1351928:  
aux_output_1351930:"
dense_24_1351934:@
dense_24_1351936:@"
dense_25_1351939:@@
dense_25_1351941:@"
dense_26_1351944:@@
dense_26_1351946:@%
main_output_1351949:@!
main_output_1351951:
identity

identity_1??"aux_output/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?6token_and_position_embedding_2/StatefulPartitionedCall?+transformer_block_2/StatefulPartitionedCall?
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_3&token_and_position_embedding_2_1351889&token_and_position_embedding_2_1351891*
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
GPU 2J 8? *d
f_R]
[__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_135076928
6token_and_position_embedding_2/StatefulPartitionedCall?
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_1351894transformer_block_2_1351896transformer_block_2_1351898transformer_block_2_1351900transformer_block_2_1351902transformer_block_2_1351904transformer_block_2_1351906transformer_block_2_1351908transformer_block_2_1351910transformer_block_2_1351912transformer_block_2_1351914transformer_block_2_1351916transformer_block_2_1351918transformer_block_2_1351920transformer_block_2_1351922transformer_block_2_1351924*
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
GPU 2J 8? *Y
fTRR
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_13510192-
+transformer_block_2/StatefulPartitionedCall?
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13510582,
*global_average_pooling1d_2/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0aux_output_1351928aux_output_1351930*
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
G__inference_aux_output_layer_call_and_return_conditional_losses_13510712$
"aux_output/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0	aux_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_13510842
concatenate_2/PartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_24_1351934dense_24_1351936*
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
GPU 2J 8? *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_13510972"
 dense_24/StatefulPartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_1351939dense_25_1351941*
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
GPU 2J 8? *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_13511142"
 dense_25/StatefulPartitionedCall?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_1351944dense_26_1351946*
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
GPU 2J 8? *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_13511312"
 dense_26/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0main_output_1351949main_output_1351951*
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
H__inference_main_output_layer_call_and_return_conditional_losses_13511482%
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
NoOpNoOp#^aux_output/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall$^main_output/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????(
!
_user_specified_name	input_3:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input
?
X
<__inference_global_average_pooling1d_2_layer_call_fn_1353470

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
GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13510582
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
??
? 
D__inference_model_2_layer_call_and_return_conditional_losses_1352532
inputs_0
inputs_1U
Ctoken_and_position_embedding_2_embedding_5_embedding_lookup_1352239:( U
Ctoken_and_position_embedding_2_embedding_4_embedding_lookup_1352245: l
Ztransformer_block_2_multi_head_self_attention_2_dense_18_tensordot_readvariableop_resource:  f
Xtransformer_block_2_multi_head_self_attention_2_dense_18_biasadd_readvariableop_resource: l
Ztransformer_block_2_multi_head_self_attention_2_dense_19_tensordot_readvariableop_resource:  f
Xtransformer_block_2_multi_head_self_attention_2_dense_19_biasadd_readvariableop_resource: l
Ztransformer_block_2_multi_head_self_attention_2_dense_20_tensordot_readvariableop_resource:  f
Xtransformer_block_2_multi_head_self_attention_2_dense_20_biasadd_readvariableop_resource: l
Ztransformer_block_2_multi_head_self_attention_2_dense_21_tensordot_readvariableop_resource:  f
Xtransformer_block_2_multi_head_self_attention_2_dense_21_biasadd_readvariableop_resource: ]
Otransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource: Y
Ktransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource: ]
Ktransformer_block_2_sequential_2_dense_22_tensordot_readvariableop_resource:  W
Itransformer_block_2_sequential_2_dense_22_biasadd_readvariableop_resource: ]
Ktransformer_block_2_sequential_2_dense_23_tensordot_readvariableop_resource:  W
Itransformer_block_2_sequential_2_dense_23_biasadd_readvariableop_resource: ]
Otransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource: Y
Ktransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource: ;
)aux_output_matmul_readvariableop_resource: 8
*aux_output_biasadd_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource:@6
(dense_24_biasadd_readvariableop_resource:@9
'dense_25_matmul_readvariableop_resource:@@6
(dense_25_biasadd_readvariableop_resource:@9
'dense_26_matmul_readvariableop_resource:@@6
(dense_26_biasadd_readvariableop_resource:@<
*main_output_matmul_readvariableop_resource:@9
+main_output_biasadd_readvariableop_resource:
identity

identity_1??!aux_output/BiasAdd/ReadVariableOp? aux_output/MatMul/ReadVariableOp?dense_24/BiasAdd/ReadVariableOp?dense_24/MatMul/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?"main_output/BiasAdd/ReadVariableOp?!main_output/MatMul/ReadVariableOp?;token_and_position_embedding_2/embedding_4/embedding_lookup?;token_and_position_embedding_2/embedding_5/embedding_lookup?Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp?Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp?Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp?Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp?Otransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?Otransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?Otransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?Otransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?@transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp?Btransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp?@transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp?Btransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp?
$token_and_position_embedding_2/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_2/Shape?
2token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2token_and_position_embedding_2/strided_slice/stack?
4token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_2/strided_slice/stack_1?
4token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_2/strided_slice/stack_2?
,token_and_position_embedding_2/strided_sliceStridedSlice-token_and_position_embedding_2/Shape:output:0;token_and_position_embedding_2/strided_slice/stack:output:0=token_and_position_embedding_2/strided_slice/stack_1:output:0=token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_2/strided_slice?
*token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_2/range/start?
*token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_2/range/delta?
$token_and_position_embedding_2/rangeRange3token_and_position_embedding_2/range/start:output:05token_and_position_embedding_2/strided_slice:output:03token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:?????????2&
$token_and_position_embedding_2/range?
;token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherCtoken_and_position_embedding_2_embedding_5_embedding_lookup_1352239-token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/1352239*'
_output_shapes
:????????? *
dtype02=
;token_and_position_embedding_2/embedding_5/embedding_lookup?
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@token_and_position_embedding_2/embedding_5/embedding_lookup/1352239*'
_output_shapes
:????????? 2F
Dtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity?
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2H
Ftoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1?
/token_and_position_embedding_2/embedding_4/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????(21
/token_and_position_embedding_2/embedding_4/Cast?
;token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherCtoken_and_position_embedding_2_embedding_4_embedding_lookup_13522453token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*V
_classL
JHloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/1352245*+
_output_shapes
:?????????( *
dtype02=
;token_and_position_embedding_2/embedding_4/embedding_lookup?
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityDtoken_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@token_and_position_embedding_2/embedding_4/embedding_lookup/1352245*+
_output_shapes
:?????????( 2F
Dtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity?
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityMtoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2H
Ftoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1?
"token_and_position_embedding_2/addAddV2Otoken_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Otoken_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2$
"token_and_position_embedding_2/add?
5transformer_block_2/multi_head_self_attention_2/ShapeShape&token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:27
5transformer_block_2/multi_head_self_attention_2/Shape?
Ctransformer_block_2/multi_head_self_attention_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_2/multi_head_self_attention_2/strided_slice/stack?
Etransformer_block_2/multi_head_self_attention_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Etransformer_block_2/multi_head_self_attention_2/strided_slice/stack_1?
Etransformer_block_2/multi_head_self_attention_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Etransformer_block_2/multi_head_self_attention_2/strided_slice/stack_2?
=transformer_block_2/multi_head_self_attention_2/strided_sliceStridedSlice>transformer_block_2/multi_head_self_attention_2/Shape:output:0Ltransformer_block_2/multi_head_self_attention_2/strided_slice/stack:output:0Ntransformer_block_2/multi_head_self_attention_2/strided_slice/stack_1:output:0Ntransformer_block_2/multi_head_self_attention_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=transformer_block_2/multi_head_self_attention_2/strided_slice?
Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_2_multi_head_self_attention_2_dense_18_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axes?
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/free?
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ShapeShape&token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Shape?
Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/free:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2?
Rtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis?
Mtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axes:output:0[transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1?
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const?
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ProdProdTtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod?
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_1?
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod_1ProdVtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1:output:0Stransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod_1?
Ntransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat/axis?
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concatConcatV2Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/free:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axes:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat?
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/stackPackPtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod:output:0Rtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/stack?
Ltransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/transpose	Transpose&token_and_position_embedding_2/add:z:0Rtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/transpose?
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReshapeReshapePtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/transpose:y:0Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Reshape?
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/MatMulMatMulStransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Reshape:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/MatMul?
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_2?
Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1ConcatV2Ttransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0Stransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_2:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1?
Btransformer_block_2/multi_head_self_attention_2/dense_18/TensordotReshapeStransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/MatMul:product:0Ttransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot?
Otransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_2_multi_head_self_attention_2_dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?
@transformer_block_2/multi_head_self_attention_2/dense_18/BiasAddBiasAddKtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd?
Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_2_multi_head_self_attention_2_dense_19_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axes?
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/free?
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ShapeShape&token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Shape?
Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/free:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2?
Rtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis?
Mtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axes:output:0[transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1?
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const?
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ProdProdTtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod?
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_1?
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod_1ProdVtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1:output:0Stransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod_1?
Ntransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat/axis?
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concatConcatV2Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/free:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axes:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat?
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/stackPackPtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod:output:0Rtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/stack?
Ltransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/transpose	Transpose&token_and_position_embedding_2/add:z:0Rtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/transpose?
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReshapeReshapePtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/transpose:y:0Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Reshape?
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/MatMulMatMulStransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Reshape:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/MatMul?
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_2?
Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1ConcatV2Ttransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0Stransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_2:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1?
Btransformer_block_2/multi_head_self_attention_2/dense_19/TensordotReshapeStransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/MatMul:product:0Ttransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot?
Otransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_2_multi_head_self_attention_2_dense_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?
@transformer_block_2/multi_head_self_attention_2/dense_19/BiasAddBiasAddKtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd?
Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_2_multi_head_self_attention_2_dense_20_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axes?
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/free?
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ShapeShape&token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Shape?
Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/free:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2?
Rtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis?
Mtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axes:output:0[transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1?
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const?
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ProdProdTtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod?
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_1?
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod_1ProdVtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1:output:0Stransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod_1?
Ntransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat/axis?
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concatConcatV2Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/free:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axes:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat?
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/stackPackPtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod:output:0Rtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/stack?
Ltransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/transpose	Transpose&token_and_position_embedding_2/add:z:0Rtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/transpose?
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReshapeReshapePtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/transpose:y:0Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Reshape?
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/MatMulMatMulStransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Reshape:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/MatMul?
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_2?
Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1ConcatV2Ttransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0Stransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_2:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1?
Btransformer_block_2/multi_head_self_attention_2/dense_20/TensordotReshapeStransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/MatMul:product:0Ttransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot?
Otransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_2_multi_head_self_attention_2_dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?
@transformer_block_2/multi_head_self_attention_2/dense_20/BiasAddBiasAddKtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd?
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2A
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/1?
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2A
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/2?
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2A
?transformer_block_2/multi_head_self_attention_2/Reshape/shape/3?
=transformer_block_2/multi_head_self_attention_2/Reshape/shapePackFtransformer_block_2/multi_head_self_attention_2/strided_slice:output:0Htransformer_block_2/multi_head_self_attention_2/Reshape/shape/1:output:0Htransformer_block_2/multi_head_self_attention_2/Reshape/shape/2:output:0Htransformer_block_2/multi_head_self_attention_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_2/multi_head_self_attention_2/Reshape/shape?
7transformer_block_2/multi_head_self_attention_2/ReshapeReshapeItransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd:output:0Ftransformer_block_2/multi_head_self_attention_2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????29
7transformer_block_2/multi_head_self_attention_2/Reshape?
>transformer_block_2/multi_head_self_attention_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2@
>transformer_block_2/multi_head_self_attention_2/transpose/perm?
9transformer_block_2/multi_head_self_attention_2/transpose	Transpose@transformer_block_2/multi_head_self_attention_2/Reshape:output:0Gtransformer_block_2/multi_head_self_attention_2/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_2/multi_head_self_attention_2/transpose?
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/1?
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/2?
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/3?
?transformer_block_2/multi_head_self_attention_2/Reshape_1/shapePackFtransformer_block_2/multi_head_self_attention_2/strided_slice:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/1:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/2:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_2/multi_head_self_attention_2/Reshape_1/shape?
9transformer_block_2/multi_head_self_attention_2/Reshape_1ReshapeItransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd:output:0Htransformer_block_2/multi_head_self_attention_2/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_2/multi_head_self_attention_2/Reshape_1?
@transformer_block_2/multi_head_self_attention_2/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_2/multi_head_self_attention_2/transpose_1/perm?
;transformer_block_2/multi_head_self_attention_2/transpose_1	TransposeBtransformer_block_2/multi_head_self_attention_2/Reshape_1:output:0Itransformer_block_2/multi_head_self_attention_2/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_2/multi_head_self_attention_2/transpose_1?
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/1?
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/2?
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/3?
?transformer_block_2/multi_head_self_attention_2/Reshape_2/shapePackFtransformer_block_2/multi_head_self_attention_2/strided_slice:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/1:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/2:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_2/multi_head_self_attention_2/Reshape_2/shape?
9transformer_block_2/multi_head_self_attention_2/Reshape_2ReshapeItransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd:output:0Htransformer_block_2/multi_head_self_attention_2/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_2/multi_head_self_attention_2/Reshape_2?
@transformer_block_2/multi_head_self_attention_2/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_2/multi_head_self_attention_2/transpose_2/perm?
;transformer_block_2/multi_head_self_attention_2/transpose_2	TransposeBtransformer_block_2/multi_head_self_attention_2/Reshape_2:output:0Itransformer_block_2/multi_head_self_attention_2/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_2/multi_head_self_attention_2/transpose_2?
6transformer_block_2/multi_head_self_attention_2/MatMulBatchMatMulV2=transformer_block_2/multi_head_self_attention_2/transpose:y:0?transformer_block_2/multi_head_self_attention_2/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(28
6transformer_block_2/multi_head_self_attention_2/MatMul?
7transformer_block_2/multi_head_self_attention_2/Shape_1Shape?transformer_block_2/multi_head_self_attention_2/transpose_1:y:0*
T0*
_output_shapes
:29
7transformer_block_2/multi_head_self_attention_2/Shape_1?
Etransformer_block_2/multi_head_self_attention_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2G
Etransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack?
Gtransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gtransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_1?
Gtransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_2?
?transformer_block_2/multi_head_self_attention_2/strided_slice_1StridedSlice@transformer_block_2/multi_head_self_attention_2/Shape_1:output:0Ntransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack:output:0Ptransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_1:output:0Ptransformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?transformer_block_2/multi_head_self_attention_2/strided_slice_1?
4transformer_block_2/multi_head_self_attention_2/CastCastHtransformer_block_2/multi_head_self_attention_2/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 26
4transformer_block_2/multi_head_self_attention_2/Cast?
4transformer_block_2/multi_head_self_attention_2/SqrtSqrt8transformer_block_2/multi_head_self_attention_2/Cast:y:0*
T0*
_output_shapes
: 26
4transformer_block_2/multi_head_self_attention_2/Sqrt?
7transformer_block_2/multi_head_self_attention_2/truedivRealDiv?transformer_block_2/multi_head_self_attention_2/MatMul:output:08transformer_block_2/multi_head_self_attention_2/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????29
7transformer_block_2/multi_head_self_attention_2/truediv?
7transformer_block_2/multi_head_self_attention_2/SoftmaxSoftmax;transformer_block_2/multi_head_self_attention_2/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????29
7transformer_block_2/multi_head_self_attention_2/Softmax?
8transformer_block_2/multi_head_self_attention_2/MatMul_1BatchMatMulV2Atransformer_block_2/multi_head_self_attention_2/Softmax:softmax:0?transformer_block_2/multi_head_self_attention_2/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2:
8transformer_block_2/multi_head_self_attention_2/MatMul_1?
@transformer_block_2/multi_head_self_attention_2/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_2/multi_head_self_attention_2/transpose_3/perm?
;transformer_block_2/multi_head_self_attention_2/transpose_3	TransposeAtransformer_block_2/multi_head_self_attention_2/MatMul_1:output:0Itransformer_block_2/multi_head_self_attention_2/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_2/multi_head_self_attention_2/transpose_3?
Atransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/1?
Atransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/2?
?transformer_block_2/multi_head_self_attention_2/Reshape_3/shapePackFtransformer_block_2/multi_head_self_attention_2/strided_slice:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/1:output:0Jtransformer_block_2/multi_head_self_attention_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_2/multi_head_self_attention_2/Reshape_3/shape?
9transformer_block_2/multi_head_self_attention_2/Reshape_3Reshape?transformer_block_2/multi_head_self_attention_2/transpose_3:y:0Htransformer_block_2/multi_head_self_attention_2/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2;
9transformer_block_2/multi_head_self_attention_2/Reshape_3?
Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_2_multi_head_self_attention_2_dense_21_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axes?
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/free?
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ShapeShapeBtransformer_block_2/multi_head_self_attention_2/Reshape_3:output:0*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Shape?
Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/free:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2?
Rtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis?
Mtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1GatherV2Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axes:output:0[transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1?
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const?
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ProdProdTtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod?
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_1?
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod_1ProdVtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1:output:0Stransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod_1?
Ntransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat/axis?
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concatConcatV2Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/free:output:0Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axes:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat?
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/stackPackPtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod:output:0Rtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/stack?
Ltransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/transpose	TransposeBtransformer_block_2/multi_head_self_attention_2/Reshape_3:output:0Rtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2N
Ltransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/transpose?
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReshapeReshapePtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/transpose:y:0Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Reshape?
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/MatMulMatMulStransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Reshape:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/MatMul?
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_2?
Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1/axis?
Ktransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1ConcatV2Ttransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0Stransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_2:output:0Ytransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1?
Btransformer_block_2/multi_head_self_attention_2/dense_21/TensordotReshapeStransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/MatMul:product:0Ttransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 2D
Btransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot?
Otransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_2_multi_head_self_attention_2_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?
@transformer_block_2/multi_head_self_attention_2/dense_21/BiasAddBiasAddKtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot:output:0Wtransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2B
@transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd?
&transformer_block_2/dropout_4/IdentityIdentityItransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 2(
&transformer_block_2/dropout_4/Identity?
transformer_block_2/addAddV2&token_and_position_embedding_2/add:z:0/transformer_block_2/dropout_4/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_2/add?
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_4/moments/mean/reduction_indices?
6transformer_block_2/layer_normalization_4/moments/meanMeantransformer_block_2/add:z:0Qtransformer_block_2/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(28
6transformer_block_2/layer_normalization_4/moments/mean?
>transformer_block_2/layer_normalization_4/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2@
>transformer_block_2/layer_normalization_4/moments/StopGradient?
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add:z:0Gtransformer_block_2/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2E
Ctransformer_block_2/layer_normalization_4/moments/SquaredDifference?
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_4/moments/variance/reduction_indices?
:transformer_block_2/layer_normalization_4/moments/varianceMeanGtransformer_block_2/layer_normalization_4/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2<
:transformer_block_2/layer_normalization_4/moments/variance?
9transformer_block_2/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_2/layer_normalization_4/batchnorm/add/y?
7transformer_block_2/layer_normalization_4/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_4/moments/variance:output:0Btransformer_block_2/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(29
7transformer_block_2/layer_normalization_4/batchnorm/add?
9transformer_block_2/layer_normalization_4/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2;
9transformer_block_2/layer_normalization_4/batchnorm/Rsqrt?
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp?
7transformer_block_2/layer_normalization_4/batchnorm/mulMul=transformer_block_2/layer_normalization_4/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 29
7transformer_block_2/layer_normalization_4/batchnorm/mul?
9transformer_block_2/layer_normalization_4/batchnorm/mul_1Multransformer_block_2/add:z:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_1?
9transformer_block_2/layer_normalization_4/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_4/moments/mean:output:0;transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_4/batchnorm/mul_2?
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp?
7transformer_block_2/layer_normalization_4/batchnorm/subSubJtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 29
7transformer_block_2/layer_normalization_4/batchnorm/sub?
9transformer_block_2/layer_normalization_4/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_4/batchnorm/add_1?
Btransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_2_sequential_2_dense_22_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02D
Btransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp?
8transformer_block_2/sequential_2/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_2/sequential_2/dense_22/Tensordot/axes?
8transformer_block_2/sequential_2/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_2/sequential_2/dense_22/Tensordot/free?
9transformer_block_2/sequential_2/dense_22/Tensordot/ShapeShape=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_22/Tensordot/Shape?
Atransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2/axis?
<transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2GatherV2Btransformer_block_2/sequential_2/dense_22/Tensordot/Shape:output:0Atransformer_block_2/sequential_2/dense_22/Tensordot/free:output:0Jtransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2?
Ctransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1/axis?
>transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1GatherV2Btransformer_block_2/sequential_2/dense_22/Tensordot/Shape:output:0Atransformer_block_2/sequential_2/dense_22/Tensordot/axes:output:0Ltransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1?
9transformer_block_2/sequential_2/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_2/sequential_2/dense_22/Tensordot/Const?
8transformer_block_2/sequential_2/dense_22/Tensordot/ProdProdEtransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2:output:0Btransformer_block_2/sequential_2/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_2/sequential_2/dense_22/Tensordot/Prod?
;transformer_block_2/sequential_2/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_2/sequential_2/dense_22/Tensordot/Const_1?
:transformer_block_2/sequential_2/dense_22/Tensordot/Prod_1ProdGtransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1:output:0Dtransformer_block_2/sequential_2/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_2/sequential_2/dense_22/Tensordot/Prod_1?
?transformer_block_2/sequential_2/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_2/sequential_2/dense_22/Tensordot/concat/axis?
:transformer_block_2/sequential_2/dense_22/Tensordot/concatConcatV2Atransformer_block_2/sequential_2/dense_22/Tensordot/free:output:0Atransformer_block_2/sequential_2/dense_22/Tensordot/axes:output:0Htransformer_block_2/sequential_2/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_2/sequential_2/dense_22/Tensordot/concat?
9transformer_block_2/sequential_2/dense_22/Tensordot/stackPackAtransformer_block_2/sequential_2/dense_22/Tensordot/Prod:output:0Ctransformer_block_2/sequential_2/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_22/Tensordot/stack?
=transformer_block_2/sequential_2/dense_22/Tensordot/transpose	Transpose=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0Ctransformer_block_2/sequential_2/dense_22/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2?
=transformer_block_2/sequential_2/dense_22/Tensordot/transpose?
;transformer_block_2/sequential_2/dense_22/Tensordot/ReshapeReshapeAtransformer_block_2/sequential_2/dense_22/Tensordot/transpose:y:0Btransformer_block_2/sequential_2/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_2/sequential_2/dense_22/Tensordot/Reshape?
:transformer_block_2/sequential_2/dense_22/Tensordot/MatMulMatMulDtransformer_block_2/sequential_2/dense_22/Tensordot/Reshape:output:0Jtransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_2/sequential_2/dense_22/Tensordot/MatMul?
;transformer_block_2/sequential_2/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_2/sequential_2/dense_22/Tensordot/Const_2?
Atransformer_block_2/sequential_2/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_2/sequential_2/dense_22/Tensordot/concat_1/axis?
<transformer_block_2/sequential_2/dense_22/Tensordot/concat_1ConcatV2Etransformer_block_2/sequential_2/dense_22/Tensordot/GatherV2:output:0Dtransformer_block_2/sequential_2/dense_22/Tensordot/Const_2:output:0Jtransformer_block_2/sequential_2/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_2/sequential_2/dense_22/Tensordot/concat_1?
3transformer_block_2/sequential_2/dense_22/TensordotReshapeDtransformer_block_2/sequential_2/dense_22/Tensordot/MatMul:product:0Etransformer_block_2/sequential_2/dense_22/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 25
3transformer_block_2/sequential_2/dense_22/Tensordot?
@transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_2_sequential_2_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp?
1transformer_block_2/sequential_2/dense_22/BiasAddBiasAdd<transformer_block_2/sequential_2/dense_22/Tensordot:output:0Htransformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 23
1transformer_block_2/sequential_2/dense_22/BiasAdd?
.transformer_block_2/sequential_2/dense_22/ReluRelu:transformer_block_2/sequential_2/dense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 20
.transformer_block_2/sequential_2/dense_22/Relu?
Btransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_2_sequential_2_dense_23_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02D
Btransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp?
8transformer_block_2/sequential_2/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_2/sequential_2/dense_23/Tensordot/axes?
8transformer_block_2/sequential_2/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_2/sequential_2/dense_23/Tensordot/free?
9transformer_block_2/sequential_2/dense_23/Tensordot/ShapeShape<transformer_block_2/sequential_2/dense_22/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_23/Tensordot/Shape?
Atransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2/axis?
<transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2GatherV2Btransformer_block_2/sequential_2/dense_23/Tensordot/Shape:output:0Atransformer_block_2/sequential_2/dense_23/Tensordot/free:output:0Jtransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2?
Ctransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1/axis?
>transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1GatherV2Btransformer_block_2/sequential_2/dense_23/Tensordot/Shape:output:0Atransformer_block_2/sequential_2/dense_23/Tensordot/axes:output:0Ltransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1?
9transformer_block_2/sequential_2/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_2/sequential_2/dense_23/Tensordot/Const?
8transformer_block_2/sequential_2/dense_23/Tensordot/ProdProdEtransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2:output:0Btransformer_block_2/sequential_2/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_2/sequential_2/dense_23/Tensordot/Prod?
;transformer_block_2/sequential_2/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_2/sequential_2/dense_23/Tensordot/Const_1?
:transformer_block_2/sequential_2/dense_23/Tensordot/Prod_1ProdGtransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1:output:0Dtransformer_block_2/sequential_2/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_2/sequential_2/dense_23/Tensordot/Prod_1?
?transformer_block_2/sequential_2/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_2/sequential_2/dense_23/Tensordot/concat/axis?
:transformer_block_2/sequential_2/dense_23/Tensordot/concatConcatV2Atransformer_block_2/sequential_2/dense_23/Tensordot/free:output:0Atransformer_block_2/sequential_2/dense_23/Tensordot/axes:output:0Htransformer_block_2/sequential_2/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_2/sequential_2/dense_23/Tensordot/concat?
9transformer_block_2/sequential_2/dense_23/Tensordot/stackPackAtransformer_block_2/sequential_2/dense_23/Tensordot/Prod:output:0Ctransformer_block_2/sequential_2/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_2/sequential_2/dense_23/Tensordot/stack?
=transformer_block_2/sequential_2/dense_23/Tensordot/transpose	Transpose<transformer_block_2/sequential_2/dense_22/Relu:activations:0Ctransformer_block_2/sequential_2/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2?
=transformer_block_2/sequential_2/dense_23/Tensordot/transpose?
;transformer_block_2/sequential_2/dense_23/Tensordot/ReshapeReshapeAtransformer_block_2/sequential_2/dense_23/Tensordot/transpose:y:0Btransformer_block_2/sequential_2/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_2/sequential_2/dense_23/Tensordot/Reshape?
:transformer_block_2/sequential_2/dense_23/Tensordot/MatMulMatMulDtransformer_block_2/sequential_2/dense_23/Tensordot/Reshape:output:0Jtransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_2/sequential_2/dense_23/Tensordot/MatMul?
;transformer_block_2/sequential_2/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_2/sequential_2/dense_23/Tensordot/Const_2?
Atransformer_block_2/sequential_2/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_2/sequential_2/dense_23/Tensordot/concat_1/axis?
<transformer_block_2/sequential_2/dense_23/Tensordot/concat_1ConcatV2Etransformer_block_2/sequential_2/dense_23/Tensordot/GatherV2:output:0Dtransformer_block_2/sequential_2/dense_23/Tensordot/Const_2:output:0Jtransformer_block_2/sequential_2/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_2/sequential_2/dense_23/Tensordot/concat_1?
3transformer_block_2/sequential_2/dense_23/TensordotReshapeDtransformer_block_2/sequential_2/dense_23/Tensordot/MatMul:product:0Etransformer_block_2/sequential_2/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 25
3transformer_block_2/sequential_2/dense_23/Tensordot?
@transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_2_sequential_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp?
1transformer_block_2/sequential_2/dense_23/BiasAddBiasAdd<transformer_block_2/sequential_2/dense_23/Tensordot:output:0Htransformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 23
1transformer_block_2/sequential_2/dense_23/BiasAdd?
&transformer_block_2/dropout_5/IdentityIdentity:transformer_block_2/sequential_2/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2(
&transformer_block_2/dropout_5/Identity?
transformer_block_2/add_1AddV2=transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0/transformer_block_2/dropout_5/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_2/add_1?
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
Htransformer_block_2/layer_normalization_5/moments/mean/reduction_indices?
6transformer_block_2/layer_normalization_5/moments/meanMeantransformer_block_2/add_1:z:0Qtransformer_block_2/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(28
6transformer_block_2/layer_normalization_5/moments/mean?
>transformer_block_2/layer_normalization_5/moments/StopGradientStopGradient?transformer_block_2/layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2@
>transformer_block_2/layer_normalization_5/moments/StopGradient?
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifferenceSquaredDifferencetransformer_block_2/add_1:z:0Gtransformer_block_2/layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2E
Ctransformer_block_2/layer_normalization_5/moments/SquaredDifference?
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
Ltransformer_block_2/layer_normalization_5/moments/variance/reduction_indices?
:transformer_block_2/layer_normalization_5/moments/varianceMeanGtransformer_block_2/layer_normalization_5/moments/SquaredDifference:z:0Utransformer_block_2/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2<
:transformer_block_2/layer_normalization_5/moments/variance?
9transformer_block_2/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52;
9transformer_block_2/layer_normalization_5/batchnorm/add/y?
7transformer_block_2/layer_normalization_5/batchnorm/addAddV2Ctransformer_block_2/layer_normalization_5/moments/variance:output:0Btransformer_block_2/layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(29
7transformer_block_2/layer_normalization_5/batchnorm/add?
9transformer_block_2/layer_normalization_5/batchnorm/RsqrtRsqrt;transformer_block_2/layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2;
9transformer_block_2/layer_normalization_5/batchnorm/Rsqrt?
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpOtransformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02H
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp?
7transformer_block_2/layer_normalization_5/batchnorm/mulMul=transformer_block_2/layer_normalization_5/batchnorm/Rsqrt:y:0Ntransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 29
7transformer_block_2/layer_normalization_5/batchnorm/mul?
9transformer_block_2/layer_normalization_5/batchnorm/mul_1Multransformer_block_2/add_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_1?
9transformer_block_2/layer_normalization_5/batchnorm/mul_2Mul?transformer_block_2/layer_normalization_5/moments/mean:output:0;transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_5/batchnorm/mul_2?
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOpKtransformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02D
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp?
7transformer_block_2/layer_normalization_5/batchnorm/subSubJtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp:value:0=transformer_block_2/layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 29
7transformer_block_2/layer_normalization_5/batchnorm/sub?
9transformer_block_2/layer_normalization_5/batchnorm/add_1AddV2=transformer_block_2/layer_normalization_5/batchnorm/mul_1:z:0;transformer_block_2/layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2;
9transformer_block_2/layer_normalization_5/batchnorm/add_1?
1global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_2/Mean/reduction_indices?
global_average_pooling1d_2/MeanMean=transformer_block_2/layer_normalization_5/batchnorm/add_1:z:0:global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2!
global_average_pooling1d_2/Mean?
 aux_output/MatMul/ReadVariableOpReadVariableOp)aux_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 aux_output/MatMul/ReadVariableOp?
aux_output/MatMulMatMul(global_average_pooling1d_2/Mean:output:0(aux_output/MatMul/ReadVariableOp:value:0*
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
aux_output/Sigmoidx
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV2aux_output/Sigmoid:y:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_2/concat?
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_24/MatMul/ReadVariableOp?
dense_24/MatMulMatMulconcatenate_2/concat:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_24/MatMul?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_24/BiasAdds
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_24/Relu?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_25/MatMul/ReadVariableOp?
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_25/MatMul?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_25/Relu?
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_26/MatMul/ReadVariableOp?
dense_26/MatMulMatMuldense_25/Relu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_26/MatMul?
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_26/BiasAdd/ReadVariableOp?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_26/Relu?
!main_output/MatMul/ReadVariableOpReadVariableOp*main_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!main_output/MatMul/ReadVariableOp?
main_output/MatMulMatMuldense_26/Relu:activations:0)main_output/MatMul/ReadVariableOp:value:0*
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

Identity_1?
NoOpNoOp"^aux_output/BiasAdd/ReadVariableOp!^aux_output/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp#^main_output/BiasAdd/ReadVariableOp"^main_output/MatMul/ReadVariableOp<^token_and_position_embedding_2/embedding_4/embedding_lookup<^token_and_position_embedding_2/embedding_5/embedding_lookupC^transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpC^transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpG^transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpP^transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpR^transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpP^transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpR^transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpP^transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpR^transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpP^transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpR^transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpA^transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOpC^transformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOpA^transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOpC^transformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!aux_output/BiasAdd/ReadVariableOp!aux_output/BiasAdd/ReadVariableOp2D
 aux_output/MatMul/ReadVariableOp aux_output/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2H
"main_output/BiasAdd/ReadVariableOp"main_output/BiasAdd/ReadVariableOp2F
!main_output/MatMul/ReadVariableOp!main_output/MatMul/ReadVariableOp2z
;token_and_position_embedding_2/embedding_4/embedding_lookup;token_and_position_embedding_2/embedding_4/embedding_lookup2z
;token_and_position_embedding_2/embedding_5/embedding_lookup;token_and_position_embedding_2/embedding_5/embedding_lookup2?
Btransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp2?
Ftransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp2?
Btransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpBtransformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp2?
Ftransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpFtransformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp2?
Otransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpOtransformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp2?
Qtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp2?
Otransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpOtransformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp2?
Qtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp2?
Otransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpOtransformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp2?
Qtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp2?
Otransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpOtransformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp2?
Qtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpQtransformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp2?
@transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp@transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp2?
Btransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOpBtransformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp2?
@transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp@transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp2?
Btransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOpBtransformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp:Q M
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
inputs/1
?
?
.__inference_sequential_2_layer_call_fn_1353608

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
GPU 2J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_13506002
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
?
E__inference_dense_24_layer_call_and_return_conditional_losses_1353535

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1353476

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
?
?
.__inference_sequential_2_layer_call_fn_1353621

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
GPU 2J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_13506602
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
?
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1350698
dense_22_input"
dense_22_1350687:  
dense_22_1350689: "
dense_23_1350692:  
dense_23_1350694: 
identity?? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCalldense_22_inputdense_22_1350687dense_22_1350689*
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
GPU 2J 8? *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_13505572"
 dense_22/StatefulPartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1350692dense_23_1350694*
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
GPU 2J 8? *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_13505932"
 dense_23/StatefulPartitionedCall?
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:[ W
+
_output_shapes
:?????????( 
(
_user_specified_namedense_22_input
?
?
-__inference_main_output_layer_call_fn_1353584

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
H__inference_main_output_layer_call_and_return_conditional_losses_13511482
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
?
?
,__inference_aux_output_layer_call_fn_1353491

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
G__inference_aux_output_layer_call_and_return_conditional_losses_13510712
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
?K
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1353735

inputs<
*dense_22_tensordot_readvariableop_resource:  6
(dense_22_biasadd_readvariableop_resource: <
*dense_23_tensordot_readvariableop_resource:  6
(dense_23_biasadd_readvariableop_resource: 
identity??dense_22/BiasAdd/ReadVariableOp?!dense_22/Tensordot/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?!dense_23/Tensordot/ReadVariableOp?
!dense_22/Tensordot/ReadVariableOpReadVariableOp*dense_22_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_22/Tensordot/ReadVariableOp|
dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_22/Tensordot/axes?
dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_22/Tensordot/freej
dense_22/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_22/Tensordot/Shape?
 dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_22/Tensordot/GatherV2/axis?
dense_22/Tensordot/GatherV2GatherV2!dense_22/Tensordot/Shape:output:0 dense_22/Tensordot/free:output:0)dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_22/Tensordot/GatherV2?
"dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_22/Tensordot/GatherV2_1/axis?
dense_22/Tensordot/GatherV2_1GatherV2!dense_22/Tensordot/Shape:output:0 dense_22/Tensordot/axes:output:0+dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_22/Tensordot/GatherV2_1~
dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_22/Tensordot/Const?
dense_22/Tensordot/ProdProd$dense_22/Tensordot/GatherV2:output:0!dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_22/Tensordot/Prod?
dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_22/Tensordot/Const_1?
dense_22/Tensordot/Prod_1Prod&dense_22/Tensordot/GatherV2_1:output:0#dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_22/Tensordot/Prod_1?
dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_22/Tensordot/concat/axis?
dense_22/Tensordot/concatConcatV2 dense_22/Tensordot/free:output:0 dense_22/Tensordot/axes:output:0'dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/concat?
dense_22/Tensordot/stackPack dense_22/Tensordot/Prod:output:0"dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/stack?
dense_22/Tensordot/transpose	Transposeinputs"dense_22/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_22/Tensordot/transpose?
dense_22/Tensordot/ReshapeReshape dense_22/Tensordot/transpose:y:0!dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_22/Tensordot/Reshape?
dense_22/Tensordot/MatMulMatMul#dense_22/Tensordot/Reshape:output:0)dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_22/Tensordot/MatMul?
dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_22/Tensordot/Const_2?
 dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_22/Tensordot/concat_1/axis?
dense_22/Tensordot/concat_1ConcatV2$dense_22/Tensordot/GatherV2:output:0#dense_22/Tensordot/Const_2:output:0)dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/concat_1?
dense_22/TensordotReshape#dense_22/Tensordot/MatMul:product:0$dense_22/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_22/Tensordot?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_22/BiasAdd/ReadVariableOp?
dense_22/BiasAddBiasAdddense_22/Tensordot:output:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_22/BiasAddw
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dense_22/Relu?
!dense_23/Tensordot/ReadVariableOpReadVariableOp*dense_23_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_23/Tensordot/ReadVariableOp|
dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_23/Tensordot/axes?
dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_23/Tensordot/free
dense_23/Tensordot/ShapeShapedense_22/Relu:activations:0*
T0*
_output_shapes
:2
dense_23/Tensordot/Shape?
 dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/GatherV2/axis?
dense_23/Tensordot/GatherV2GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/free:output:0)dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2?
"dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_23/Tensordot/GatherV2_1/axis?
dense_23/Tensordot/GatherV2_1GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/axes:output:0+dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2_1~
dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const?
dense_23/Tensordot/ProdProd$dense_23/Tensordot/GatherV2:output:0!dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod?
dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const_1?
dense_23/Tensordot/Prod_1Prod&dense_23/Tensordot/GatherV2_1:output:0#dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod_1?
dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_23/Tensordot/concat/axis?
dense_23/Tensordot/concatConcatV2 dense_23/Tensordot/free:output:0 dense_23/Tensordot/axes:output:0'dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat?
dense_23/Tensordot/stackPack dense_23/Tensordot/Prod:output:0"dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/stack?
dense_23/Tensordot/transpose	Transposedense_22/Relu:activations:0"dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_23/Tensordot/transpose?
dense_23/Tensordot/ReshapeReshape dense_23/Tensordot/transpose:y:0!dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_23/Tensordot/Reshape?
dense_23/Tensordot/MatMulMatMul#dense_23/Tensordot/Reshape:output:0)dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_23/Tensordot/MatMul?
dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const_2?
 dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/concat_1/axis?
dense_23/Tensordot/concat_1ConcatV2$dense_23/Tensordot/GatherV2:output:0#dense_23/Tensordot/Const_2:output:0)dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat_1?
dense_23/TensordotReshape#dense_23/Tensordot/MatMul:product:0$dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_23/Tensordot?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_23/BiasAdd/ReadVariableOp?
dense_23/BiasAddBiasAdddense_23/Tensordot:output:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_23/BiasAddx
IdentityIdentitydense_23/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp ^dense_22/BiasAdd/ReadVariableOp"^dense_22/Tensordot/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp"^dense_23/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2F
!dense_22/Tensordot/ReadVariableOp!dense_22/Tensordot/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/Tensordot/ReadVariableOp!dense_23/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?!
?
E__inference_dense_22_layer_call_and_return_conditional_losses_1353775

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
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1351058

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
?
s
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1353482

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
?
X
<__inference_global_average_pooling1d_2_layer_call_fn_1353465

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
GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13507222
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
?
?
)__inference_model_2_layer_call_fn_1351217
input_3
	aux_input
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

unknown_19:@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3	aux_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_13511562
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
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????(
!
_user_specified_name	input_3:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input
??
?
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_1351019

inputsX
Fmulti_head_self_attention_2_dense_18_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_18_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_19_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_19_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_20_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_20_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_21_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_21_biasadd_readvariableop_resource: I
;layer_normalization_4_batchnorm_mul_readvariableop_resource: E
7layer_normalization_4_batchnorm_readvariableop_resource: I
7sequential_2_dense_22_tensordot_readvariableop_resource:  C
5sequential_2_dense_22_biasadd_readvariableop_resource: I
7sequential_2_dense_23_tensordot_readvariableop_resource:  C
5sequential_2_dense_23_biasadd_readvariableop_resource: I
;layer_normalization_5_batchnorm_mul_readvariableop_resource: E
7layer_normalization_5_batchnorm_readvariableop_resource: 
identity??.layer_normalization_4/batchnorm/ReadVariableOp?2layer_normalization_4/batchnorm/mul/ReadVariableOp?.layer_normalization_5/batchnorm/ReadVariableOp?2layer_normalization_5/batchnorm/mul/ReadVariableOp?;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?,sequential_2/dense_22/BiasAdd/ReadVariableOp?.sequential_2/dense_22/Tensordot/ReadVariableOp?,sequential_2/dense_23/BiasAdd/ReadVariableOp?.sequential_2/dense_23/Tensordot/ReadVariableOp|
!multi_head_self_attention_2/ShapeShapeinputs*
T0*
_output_shapes
:2#
!multi_head_self_attention_2/Shape?
/multi_head_self_attention_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention_2/strided_slice/stack?
1multi_head_self_attention_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_2/strided_slice/stack_1?
1multi_head_self_attention_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_2/strided_slice/stack_2?
)multi_head_self_attention_2/strided_sliceStridedSlice*multi_head_self_attention_2/Shape:output:08multi_head_self_attention_2/strided_slice/stack:output:0:multi_head_self_attention_2/strided_slice/stack_1:output:0:multi_head_self_attention_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention_2/strided_slice?
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_18_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_18/Tensordot/axes?
3multi_head_self_attention_2/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_18/Tensordot/free?
4multi_head_self_attention_2/dense_18/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_18/Tensordot/Shape?
<multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_18/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_18/Tensordot/free:output:0Emulti_head_self_attention_2/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_18/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_18/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_18/Tensordot/Const?
3multi_head_self_attention_2/dense_18/Tensordot/ProdProd@multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_18/Tensordot/Prod?
6multi_head_self_attention_2/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_18/Tensordot/Const_1?
5multi_head_self_attention_2/dense_18/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_18/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_18/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_18/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_18/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_18/Tensordot/free:output:0<multi_head_self_attention_2/dense_18/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_18/Tensordot/concat?
4multi_head_self_attention_2/dense_18/Tensordot/stackPack<multi_head_self_attention_2/dense_18/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_18/Tensordot/stack?
8multi_head_self_attention_2/dense_18/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_18/Tensordot/transpose?
6multi_head_self_attention_2/dense_18/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_18/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_18/Tensordot/Reshape?
5multi_head_self_attention_2/dense_18/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_18/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_18/Tensordot/MatMul?
6multi_head_self_attention_2/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_18/Tensordot/Const_2?
<multi_head_self_attention_2/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_18/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_18/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_18/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_18/Tensordot/concat_1?
.multi_head_self_attention_2/dense_18/TensordotReshape?multi_head_self_attention_2/dense_18/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_18/Tensordot?
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_18/BiasAddBiasAdd7multi_head_self_attention_2/dense_18/Tensordot:output:0Cmulti_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_18/BiasAdd?
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_19_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_19/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_19/Tensordot/axes?
3multi_head_self_attention_2/dense_19/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_19/Tensordot/free?
4multi_head_self_attention_2/dense_19/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_19/Tensordot/Shape?
<multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_19/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_19/Tensordot/free:output:0Emulti_head_self_attention_2/dense_19/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_19/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_19/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_19/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_19/Tensordot/Const?
3multi_head_self_attention_2/dense_19/Tensordot/ProdProd@multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_19/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_19/Tensordot/Prod?
6multi_head_self_attention_2/dense_19/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_19/Tensordot/Const_1?
5multi_head_self_attention_2/dense_19/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_19/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_19/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_19/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_19/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_19/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_19/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_19/Tensordot/free:output:0<multi_head_self_attention_2/dense_19/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_19/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_19/Tensordot/concat?
4multi_head_self_attention_2/dense_19/Tensordot/stackPack<multi_head_self_attention_2/dense_19/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_19/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_19/Tensordot/stack?
8multi_head_self_attention_2/dense_19/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_19/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_19/Tensordot/transpose?
6multi_head_self_attention_2/dense_19/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_19/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_19/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_19/Tensordot/Reshape?
5multi_head_self_attention_2/dense_19/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_19/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_19/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_19/Tensordot/MatMul?
6multi_head_self_attention_2/dense_19/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_19/Tensordot/Const_2?
<multi_head_self_attention_2/dense_19/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_19/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_19/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_19/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_19/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_19/Tensordot/concat_1?
.multi_head_self_attention_2/dense_19/TensordotReshape?multi_head_self_attention_2/dense_19/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_19/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_19/Tensordot?
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_19/BiasAddBiasAdd7multi_head_self_attention_2/dense_19/Tensordot:output:0Cmulti_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_19/BiasAdd?
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_20_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_20/Tensordot/axes?
3multi_head_self_attention_2/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_20/Tensordot/free?
4multi_head_self_attention_2/dense_20/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_20/Tensordot/Shape?
<multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_20/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_20/Tensordot/free:output:0Emulti_head_self_attention_2/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_20/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_20/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_20/Tensordot/Const?
3multi_head_self_attention_2/dense_20/Tensordot/ProdProd@multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_20/Tensordot/Prod?
6multi_head_self_attention_2/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_20/Tensordot/Const_1?
5multi_head_self_attention_2/dense_20/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_20/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_20/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_20/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_20/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_20/Tensordot/free:output:0<multi_head_self_attention_2/dense_20/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_20/Tensordot/concat?
4multi_head_self_attention_2/dense_20/Tensordot/stackPack<multi_head_self_attention_2/dense_20/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_20/Tensordot/stack?
8multi_head_self_attention_2/dense_20/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_20/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_20/Tensordot/transpose?
6multi_head_self_attention_2/dense_20/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_20/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_20/Tensordot/Reshape?
5multi_head_self_attention_2/dense_20/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_20/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_20/Tensordot/MatMul?
6multi_head_self_attention_2/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_20/Tensordot/Const_2?
<multi_head_self_attention_2/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_20/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_20/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_20/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_20/Tensordot/concat_1?
.multi_head_self_attention_2/dense_20/TensordotReshape?multi_head_self_attention_2/dense_20/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_20/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_20/Tensordot?
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_20/BiasAddBiasAdd7multi_head_self_attention_2/dense_20/Tensordot:output:0Cmulti_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_20/BiasAdd?
+multi_head_self_attention_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+multi_head_self_attention_2/Reshape/shape/1?
+multi_head_self_attention_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_2/Reshape/shape/2?
+multi_head_self_attention_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_2/Reshape/shape/3?
)multi_head_self_attention_2/Reshape/shapePack2multi_head_self_attention_2/strided_slice:output:04multi_head_self_attention_2/Reshape/shape/1:output:04multi_head_self_attention_2/Reshape/shape/2:output:04multi_head_self_attention_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention_2/Reshape/shape?
#multi_head_self_attention_2/ReshapeReshape5multi_head_self_attention_2/dense_18/BiasAdd:output:02multi_head_self_attention_2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2%
#multi_head_self_attention_2/Reshape?
*multi_head_self_attention_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention_2/transpose/perm?
%multi_head_self_attention_2/transpose	Transpose,multi_head_self_attention_2/Reshape:output:03multi_head_self_attention_2/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/transpose?
-multi_head_self_attention_2/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_1/shape/1?
-multi_head_self_attention_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_1/shape/2?
-multi_head_self_attention_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_1/shape/3?
+multi_head_self_attention_2/Reshape_1/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_1/shape/1:output:06multi_head_self_attention_2/Reshape_1/shape/2:output:06multi_head_self_attention_2/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_1/shape?
%multi_head_self_attention_2/Reshape_1Reshape5multi_head_self_attention_2/dense_19/BiasAdd:output:04multi_head_self_attention_2/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/Reshape_1?
,multi_head_self_attention_2/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_1/perm?
'multi_head_self_attention_2/transpose_1	Transpose.multi_head_self_attention_2/Reshape_1:output:05multi_head_self_attention_2/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_1?
-multi_head_self_attention_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_2/shape/1?
-multi_head_self_attention_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_2/shape/2?
-multi_head_self_attention_2/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_2/shape/3?
+multi_head_self_attention_2/Reshape_2/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_2/shape/1:output:06multi_head_self_attention_2/Reshape_2/shape/2:output:06multi_head_self_attention_2/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_2/shape?
%multi_head_self_attention_2/Reshape_2Reshape5multi_head_self_attention_2/dense_20/BiasAdd:output:04multi_head_self_attention_2/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/Reshape_2?
,multi_head_self_attention_2/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_2/perm?
'multi_head_self_attention_2/transpose_2	Transpose.multi_head_self_attention_2/Reshape_2:output:05multi_head_self_attention_2/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_2?
"multi_head_self_attention_2/MatMulBatchMatMulV2)multi_head_self_attention_2/transpose:y:0+multi_head_self_attention_2/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2$
"multi_head_self_attention_2/MatMul?
#multi_head_self_attention_2/Shape_1Shape+multi_head_self_attention_2/transpose_1:y:0*
T0*
_output_shapes
:2%
#multi_head_self_attention_2/Shape_1?
1multi_head_self_attention_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1multi_head_self_attention_2/strided_slice_1/stack?
3multi_head_self_attention_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention_2/strided_slice_1/stack_1?
3multi_head_self_attention_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/strided_slice_1/stack_2?
+multi_head_self_attention_2/strided_slice_1StridedSlice,multi_head_self_attention_2/Shape_1:output:0:multi_head_self_attention_2/strided_slice_1/stack:output:0<multi_head_self_attention_2/strided_slice_1/stack_1:output:0<multi_head_self_attention_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+multi_head_self_attention_2/strided_slice_1?
 multi_head_self_attention_2/CastCast4multi_head_self_attention_2/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 multi_head_self_attention_2/Cast?
 multi_head_self_attention_2/SqrtSqrt$multi_head_self_attention_2/Cast:y:0*
T0*
_output_shapes
: 2"
 multi_head_self_attention_2/Sqrt?
#multi_head_self_attention_2/truedivRealDiv+multi_head_self_attention_2/MatMul:output:0$multi_head_self_attention_2/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_2/truediv?
#multi_head_self_attention_2/SoftmaxSoftmax'multi_head_self_attention_2/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_2/Softmax?
$multi_head_self_attention_2/MatMul_1BatchMatMulV2-multi_head_self_attention_2/Softmax:softmax:0+multi_head_self_attention_2/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_2/MatMul_1?
,multi_head_self_attention_2/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_3/perm?
'multi_head_self_attention_2/transpose_3	Transpose-multi_head_self_attention_2/MatMul_1:output:05multi_head_self_attention_2/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_3?
-multi_head_self_attention_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_3/shape/1?
-multi_head_self_attention_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2/
-multi_head_self_attention_2/Reshape_3/shape/2?
+multi_head_self_attention_2/Reshape_3/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_3/shape/1:output:06multi_head_self_attention_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_3/shape?
%multi_head_self_attention_2/Reshape_3Reshape+multi_head_self_attention_2/transpose_3:y:04multi_head_self_attention_2/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2'
%multi_head_self_attention_2/Reshape_3?
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_21_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_21/Tensordot/axes?
3multi_head_self_attention_2/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_21/Tensordot/free?
4multi_head_self_attention_2/dense_21/Tensordot/ShapeShape.multi_head_self_attention_2/Reshape_3:output:0*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_21/Tensordot/Shape?
<multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_21/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_21/Tensordot/free:output:0Emulti_head_self_attention_2/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_21/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_21/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_21/Tensordot/Const?
3multi_head_self_attention_2/dense_21/Tensordot/ProdProd@multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_21/Tensordot/Prod?
6multi_head_self_attention_2/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_21/Tensordot/Const_1?
5multi_head_self_attention_2/dense_21/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_21/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_21/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_21/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_21/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_21/Tensordot/free:output:0<multi_head_self_attention_2/dense_21/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_21/Tensordot/concat?
4multi_head_self_attention_2/dense_21/Tensordot/stackPack<multi_head_self_attention_2/dense_21/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_21/Tensordot/stack?
8multi_head_self_attention_2/dense_21/Tensordot/transpose	Transpose.multi_head_self_attention_2/Reshape_3:output:0>multi_head_self_attention_2/dense_21/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2:
8multi_head_self_attention_2/dense_21/Tensordot/transpose?
6multi_head_self_attention_2/dense_21/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_21/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_21/Tensordot/Reshape?
5multi_head_self_attention_2/dense_21/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_21/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_21/Tensordot/MatMul?
6multi_head_self_attention_2/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_21/Tensordot/Const_2?
<multi_head_self_attention_2/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_21/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_21/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_21/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_21/Tensordot/concat_1?
.multi_head_self_attention_2/dense_21/TensordotReshape?multi_head_self_attention_2/dense_21/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_21/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_2/dense_21/Tensordot?
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_21/BiasAddBiasAdd7multi_head_self_attention_2/dense_21/Tensordot:output:0Cmulti_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2.
,multi_head_self_attention_2/dense_21/BiasAdd?
dropout_4/IdentityIdentity5multi_head_self_attention_2/dense_21/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_4/Identityn
addAddV2inputsdropout_4/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add?
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indices?
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2$
"layer_normalization_4/moments/mean?
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2,
*layer_normalization_4/moments/StopGradient?
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 21
/layer_normalization_4/moments/SquaredDifference?
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indices?
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2(
&layer_normalization_4/moments/variance?
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_4/batchnorm/add/y?
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2%
#layer_normalization_4/batchnorm/add?
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2'
%layer_normalization_4/batchnorm/Rsqrt?
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOp?
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_4/batchnorm/mul?
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/mul_1?
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/mul_2?
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOp?
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_4/batchnorm/sub?
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/add_1?
.sequential_2/dense_22/Tensordot/ReadVariableOpReadVariableOp7sequential_2_dense_22_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_2/dense_22/Tensordot/ReadVariableOp?
$sequential_2/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_2/dense_22/Tensordot/axes?
$sequential_2/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_2/dense_22/Tensordot/free?
%sequential_2/dense_22/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_2/dense_22/Tensordot/Shape?
-sequential_2/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_22/Tensordot/GatherV2/axis?
(sequential_2/dense_22/Tensordot/GatherV2GatherV2.sequential_2/dense_22/Tensordot/Shape:output:0-sequential_2/dense_22/Tensordot/free:output:06sequential_2/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_2/dense_22/Tensordot/GatherV2?
/sequential_2/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_2/dense_22/Tensordot/GatherV2_1/axis?
*sequential_2/dense_22/Tensordot/GatherV2_1GatherV2.sequential_2/dense_22/Tensordot/Shape:output:0-sequential_2/dense_22/Tensordot/axes:output:08sequential_2/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_2/dense_22/Tensordot/GatherV2_1?
%sequential_2/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_2/dense_22/Tensordot/Const?
$sequential_2/dense_22/Tensordot/ProdProd1sequential_2/dense_22/Tensordot/GatherV2:output:0.sequential_2/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_2/dense_22/Tensordot/Prod?
'sequential_2/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_22/Tensordot/Const_1?
&sequential_2/dense_22/Tensordot/Prod_1Prod3sequential_2/dense_22/Tensordot/GatherV2_1:output:00sequential_2/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_2/dense_22/Tensordot/Prod_1?
+sequential_2/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_2/dense_22/Tensordot/concat/axis?
&sequential_2/dense_22/Tensordot/concatConcatV2-sequential_2/dense_22/Tensordot/free:output:0-sequential_2/dense_22/Tensordot/axes:output:04sequential_2/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_2/dense_22/Tensordot/concat?
%sequential_2/dense_22/Tensordot/stackPack-sequential_2/dense_22/Tensordot/Prod:output:0/sequential_2/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_22/Tensordot/stack?
)sequential_2/dense_22/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0/sequential_2/dense_22/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_2/dense_22/Tensordot/transpose?
'sequential_2/dense_22/Tensordot/ReshapeReshape-sequential_2/dense_22/Tensordot/transpose:y:0.sequential_2/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_2/dense_22/Tensordot/Reshape?
&sequential_2/dense_22/Tensordot/MatMulMatMul0sequential_2/dense_22/Tensordot/Reshape:output:06sequential_2/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_2/dense_22/Tensordot/MatMul?
'sequential_2/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_22/Tensordot/Const_2?
-sequential_2/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_22/Tensordot/concat_1/axis?
(sequential_2/dense_22/Tensordot/concat_1ConcatV21sequential_2/dense_22/Tensordot/GatherV2:output:00sequential_2/dense_22/Tensordot/Const_2:output:06sequential_2/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_2/dense_22/Tensordot/concat_1?
sequential_2/dense_22/TensordotReshape0sequential_2/dense_22/Tensordot/MatMul:product:01sequential_2/dense_22/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_2/dense_22/Tensordot?
,sequential_2/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/dense_22/BiasAdd/ReadVariableOp?
sequential_2/dense_22/BiasAddBiasAdd(sequential_2/dense_22/Tensordot:output:04sequential_2/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_22/BiasAdd?
sequential_2/dense_22/ReluRelu&sequential_2/dense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_22/Relu?
.sequential_2/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_2_dense_23_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_2/dense_23/Tensordot/ReadVariableOp?
$sequential_2/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_2/dense_23/Tensordot/axes?
$sequential_2/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_2/dense_23/Tensordot/free?
%sequential_2/dense_23/Tensordot/ShapeShape(sequential_2/dense_22/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_2/dense_23/Tensordot/Shape?
-sequential_2/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_23/Tensordot/GatherV2/axis?
(sequential_2/dense_23/Tensordot/GatherV2GatherV2.sequential_2/dense_23/Tensordot/Shape:output:0-sequential_2/dense_23/Tensordot/free:output:06sequential_2/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_2/dense_23/Tensordot/GatherV2?
/sequential_2/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_2/dense_23/Tensordot/GatherV2_1/axis?
*sequential_2/dense_23/Tensordot/GatherV2_1GatherV2.sequential_2/dense_23/Tensordot/Shape:output:0-sequential_2/dense_23/Tensordot/axes:output:08sequential_2/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_2/dense_23/Tensordot/GatherV2_1?
%sequential_2/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_2/dense_23/Tensordot/Const?
$sequential_2/dense_23/Tensordot/ProdProd1sequential_2/dense_23/Tensordot/GatherV2:output:0.sequential_2/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_2/dense_23/Tensordot/Prod?
'sequential_2/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_23/Tensordot/Const_1?
&sequential_2/dense_23/Tensordot/Prod_1Prod3sequential_2/dense_23/Tensordot/GatherV2_1:output:00sequential_2/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_2/dense_23/Tensordot/Prod_1?
+sequential_2/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_2/dense_23/Tensordot/concat/axis?
&sequential_2/dense_23/Tensordot/concatConcatV2-sequential_2/dense_23/Tensordot/free:output:0-sequential_2/dense_23/Tensordot/axes:output:04sequential_2/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_2/dense_23/Tensordot/concat?
%sequential_2/dense_23/Tensordot/stackPack-sequential_2/dense_23/Tensordot/Prod:output:0/sequential_2/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_23/Tensordot/stack?
)sequential_2/dense_23/Tensordot/transpose	Transpose(sequential_2/dense_22/Relu:activations:0/sequential_2/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_2/dense_23/Tensordot/transpose?
'sequential_2/dense_23/Tensordot/ReshapeReshape-sequential_2/dense_23/Tensordot/transpose:y:0.sequential_2/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_2/dense_23/Tensordot/Reshape?
&sequential_2/dense_23/Tensordot/MatMulMatMul0sequential_2/dense_23/Tensordot/Reshape:output:06sequential_2/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_2/dense_23/Tensordot/MatMul?
'sequential_2/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_23/Tensordot/Const_2?
-sequential_2/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_23/Tensordot/concat_1/axis?
(sequential_2/dense_23/Tensordot/concat_1ConcatV21sequential_2/dense_23/Tensordot/GatherV2:output:00sequential_2/dense_23/Tensordot/Const_2:output:06sequential_2/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_2/dense_23/Tensordot/concat_1?
sequential_2/dense_23/TensordotReshape0sequential_2/dense_23/Tensordot/MatMul:product:01sequential_2/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_2/dense_23/Tensordot?
,sequential_2/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/dense_23/BiasAdd/ReadVariableOp?
sequential_2/dense_23/BiasAddBiasAdd(sequential_2/dense_23/Tensordot:output:04sequential_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_23/BiasAdd?
dropout_5/IdentityIdentity&sequential_2/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_5/Identity?
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_5/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add_1?
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indices?
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2$
"layer_normalization_5/moments/mean?
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2,
*layer_normalization_5/moments/StopGradient?
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 21
/layer_normalization_5/moments/SquaredDifference?
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indices?
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2(
&layer_normalization_5/moments/variance?
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_5/batchnorm/add/y?
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2%
#layer_normalization_5/batchnorm/add?
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2'
%layer_normalization_5/batchnorm/Rsqrt?
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOp?
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_5/batchnorm/mul?
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/mul_1?
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/mul_2?
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOp?
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_5/batchnorm/sub?
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/add_1?
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp<^multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp-^sequential_2/dense_22/BiasAdd/ReadVariableOp/^sequential_2/dense_22/Tensordot/ReadVariableOp-^sequential_2/dense_23/BiasAdd/ReadVariableOp/^sequential_2/dense_23/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2z
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp2\
,sequential_2/dense_22/BiasAdd/ReadVariableOp,sequential_2/dense_22/BiasAdd/ReadVariableOp2`
.sequential_2/dense_22/Tensordot/ReadVariableOp.sequential_2/dense_22/Tensordot/ReadVariableOp2\
,sequential_2/dense_23/BiasAdd/ReadVariableOp,sequential_2/dense_23/BiasAdd/ReadVariableOp2`
.sequential_2/dense_23/Tensordot/ReadVariableOp.sequential_2/dense_23/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
??
?
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_1353202

inputsX
Fmulti_head_self_attention_2_dense_18_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_18_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_19_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_19_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_20_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_20_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_21_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_21_biasadd_readvariableop_resource: I
;layer_normalization_4_batchnorm_mul_readvariableop_resource: E
7layer_normalization_4_batchnorm_readvariableop_resource: I
7sequential_2_dense_22_tensordot_readvariableop_resource:  C
5sequential_2_dense_22_biasadd_readvariableop_resource: I
7sequential_2_dense_23_tensordot_readvariableop_resource:  C
5sequential_2_dense_23_biasadd_readvariableop_resource: I
;layer_normalization_5_batchnorm_mul_readvariableop_resource: E
7layer_normalization_5_batchnorm_readvariableop_resource: 
identity??.layer_normalization_4/batchnorm/ReadVariableOp?2layer_normalization_4/batchnorm/mul/ReadVariableOp?.layer_normalization_5/batchnorm/ReadVariableOp?2layer_normalization_5/batchnorm/mul/ReadVariableOp?;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?,sequential_2/dense_22/BiasAdd/ReadVariableOp?.sequential_2/dense_22/Tensordot/ReadVariableOp?,sequential_2/dense_23/BiasAdd/ReadVariableOp?.sequential_2/dense_23/Tensordot/ReadVariableOp|
!multi_head_self_attention_2/ShapeShapeinputs*
T0*
_output_shapes
:2#
!multi_head_self_attention_2/Shape?
/multi_head_self_attention_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention_2/strided_slice/stack?
1multi_head_self_attention_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_2/strided_slice/stack_1?
1multi_head_self_attention_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_2/strided_slice/stack_2?
)multi_head_self_attention_2/strided_sliceStridedSlice*multi_head_self_attention_2/Shape:output:08multi_head_self_attention_2/strided_slice/stack:output:0:multi_head_self_attention_2/strided_slice/stack_1:output:0:multi_head_self_attention_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention_2/strided_slice?
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_18_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_18/Tensordot/axes?
3multi_head_self_attention_2/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_18/Tensordot/free?
4multi_head_self_attention_2/dense_18/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_18/Tensordot/Shape?
<multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_18/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_18/Tensordot/free:output:0Emulti_head_self_attention_2/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_18/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_18/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_18/Tensordot/Const?
3multi_head_self_attention_2/dense_18/Tensordot/ProdProd@multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_18/Tensordot/Prod?
6multi_head_self_attention_2/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_18/Tensordot/Const_1?
5multi_head_self_attention_2/dense_18/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_18/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_18/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_18/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_18/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_18/Tensordot/free:output:0<multi_head_self_attention_2/dense_18/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_18/Tensordot/concat?
4multi_head_self_attention_2/dense_18/Tensordot/stackPack<multi_head_self_attention_2/dense_18/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_18/Tensordot/stack?
8multi_head_self_attention_2/dense_18/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_18/Tensordot/transpose?
6multi_head_self_attention_2/dense_18/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_18/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_18/Tensordot/Reshape?
5multi_head_self_attention_2/dense_18/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_18/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_18/Tensordot/MatMul?
6multi_head_self_attention_2/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_18/Tensordot/Const_2?
<multi_head_self_attention_2/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_18/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_18/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_18/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_18/Tensordot/concat_1?
.multi_head_self_attention_2/dense_18/TensordotReshape?multi_head_self_attention_2/dense_18/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_18/Tensordot?
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_18/BiasAddBiasAdd7multi_head_self_attention_2/dense_18/Tensordot:output:0Cmulti_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_18/BiasAdd?
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_19_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_19/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_19/Tensordot/axes?
3multi_head_self_attention_2/dense_19/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_19/Tensordot/free?
4multi_head_self_attention_2/dense_19/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_19/Tensordot/Shape?
<multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_19/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_19/Tensordot/free:output:0Emulti_head_self_attention_2/dense_19/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_19/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_19/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_19/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_19/Tensordot/Const?
3multi_head_self_attention_2/dense_19/Tensordot/ProdProd@multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_19/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_19/Tensordot/Prod?
6multi_head_self_attention_2/dense_19/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_19/Tensordot/Const_1?
5multi_head_self_attention_2/dense_19/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_19/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_19/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_19/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_19/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_19/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_19/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_19/Tensordot/free:output:0<multi_head_self_attention_2/dense_19/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_19/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_19/Tensordot/concat?
4multi_head_self_attention_2/dense_19/Tensordot/stackPack<multi_head_self_attention_2/dense_19/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_19/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_19/Tensordot/stack?
8multi_head_self_attention_2/dense_19/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_19/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_19/Tensordot/transpose?
6multi_head_self_attention_2/dense_19/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_19/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_19/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_19/Tensordot/Reshape?
5multi_head_self_attention_2/dense_19/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_19/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_19/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_19/Tensordot/MatMul?
6multi_head_self_attention_2/dense_19/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_19/Tensordot/Const_2?
<multi_head_self_attention_2/dense_19/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_19/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_19/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_19/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_19/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_19/Tensordot/concat_1?
.multi_head_self_attention_2/dense_19/TensordotReshape?multi_head_self_attention_2/dense_19/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_19/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_19/Tensordot?
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_19/BiasAddBiasAdd7multi_head_self_attention_2/dense_19/Tensordot:output:0Cmulti_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_19/BiasAdd?
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_20_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_20/Tensordot/axes?
3multi_head_self_attention_2/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_20/Tensordot/free?
4multi_head_self_attention_2/dense_20/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_20/Tensordot/Shape?
<multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_20/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_20/Tensordot/free:output:0Emulti_head_self_attention_2/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_20/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_20/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_20/Tensordot/Const?
3multi_head_self_attention_2/dense_20/Tensordot/ProdProd@multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_20/Tensordot/Prod?
6multi_head_self_attention_2/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_20/Tensordot/Const_1?
5multi_head_self_attention_2/dense_20/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_20/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_20/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_20/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_20/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_20/Tensordot/free:output:0<multi_head_self_attention_2/dense_20/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_20/Tensordot/concat?
4multi_head_self_attention_2/dense_20/Tensordot/stackPack<multi_head_self_attention_2/dense_20/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_20/Tensordot/stack?
8multi_head_self_attention_2/dense_20/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_20/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_20/Tensordot/transpose?
6multi_head_self_attention_2/dense_20/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_20/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_20/Tensordot/Reshape?
5multi_head_self_attention_2/dense_20/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_20/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_20/Tensordot/MatMul?
6multi_head_self_attention_2/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_20/Tensordot/Const_2?
<multi_head_self_attention_2/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_20/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_20/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_20/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_20/Tensordot/concat_1?
.multi_head_self_attention_2/dense_20/TensordotReshape?multi_head_self_attention_2/dense_20/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_20/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_20/Tensordot?
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_20/BiasAddBiasAdd7multi_head_self_attention_2/dense_20/Tensordot:output:0Cmulti_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_20/BiasAdd?
+multi_head_self_attention_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+multi_head_self_attention_2/Reshape/shape/1?
+multi_head_self_attention_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_2/Reshape/shape/2?
+multi_head_self_attention_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_2/Reshape/shape/3?
)multi_head_self_attention_2/Reshape/shapePack2multi_head_self_attention_2/strided_slice:output:04multi_head_self_attention_2/Reshape/shape/1:output:04multi_head_self_attention_2/Reshape/shape/2:output:04multi_head_self_attention_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention_2/Reshape/shape?
#multi_head_self_attention_2/ReshapeReshape5multi_head_self_attention_2/dense_18/BiasAdd:output:02multi_head_self_attention_2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2%
#multi_head_self_attention_2/Reshape?
*multi_head_self_attention_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention_2/transpose/perm?
%multi_head_self_attention_2/transpose	Transpose,multi_head_self_attention_2/Reshape:output:03multi_head_self_attention_2/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/transpose?
-multi_head_self_attention_2/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_1/shape/1?
-multi_head_self_attention_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_1/shape/2?
-multi_head_self_attention_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_1/shape/3?
+multi_head_self_attention_2/Reshape_1/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_1/shape/1:output:06multi_head_self_attention_2/Reshape_1/shape/2:output:06multi_head_self_attention_2/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_1/shape?
%multi_head_self_attention_2/Reshape_1Reshape5multi_head_self_attention_2/dense_19/BiasAdd:output:04multi_head_self_attention_2/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/Reshape_1?
,multi_head_self_attention_2/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_1/perm?
'multi_head_self_attention_2/transpose_1	Transpose.multi_head_self_attention_2/Reshape_1:output:05multi_head_self_attention_2/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_1?
-multi_head_self_attention_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_2/shape/1?
-multi_head_self_attention_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_2/shape/2?
-multi_head_self_attention_2/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_2/shape/3?
+multi_head_self_attention_2/Reshape_2/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_2/shape/1:output:06multi_head_self_attention_2/Reshape_2/shape/2:output:06multi_head_self_attention_2/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_2/shape?
%multi_head_self_attention_2/Reshape_2Reshape5multi_head_self_attention_2/dense_20/BiasAdd:output:04multi_head_self_attention_2/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/Reshape_2?
,multi_head_self_attention_2/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_2/perm?
'multi_head_self_attention_2/transpose_2	Transpose.multi_head_self_attention_2/Reshape_2:output:05multi_head_self_attention_2/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_2?
"multi_head_self_attention_2/MatMulBatchMatMulV2)multi_head_self_attention_2/transpose:y:0+multi_head_self_attention_2/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2$
"multi_head_self_attention_2/MatMul?
#multi_head_self_attention_2/Shape_1Shape+multi_head_self_attention_2/transpose_1:y:0*
T0*
_output_shapes
:2%
#multi_head_self_attention_2/Shape_1?
1multi_head_self_attention_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1multi_head_self_attention_2/strided_slice_1/stack?
3multi_head_self_attention_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention_2/strided_slice_1/stack_1?
3multi_head_self_attention_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/strided_slice_1/stack_2?
+multi_head_self_attention_2/strided_slice_1StridedSlice,multi_head_self_attention_2/Shape_1:output:0:multi_head_self_attention_2/strided_slice_1/stack:output:0<multi_head_self_attention_2/strided_slice_1/stack_1:output:0<multi_head_self_attention_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+multi_head_self_attention_2/strided_slice_1?
 multi_head_self_attention_2/CastCast4multi_head_self_attention_2/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 multi_head_self_attention_2/Cast?
 multi_head_self_attention_2/SqrtSqrt$multi_head_self_attention_2/Cast:y:0*
T0*
_output_shapes
: 2"
 multi_head_self_attention_2/Sqrt?
#multi_head_self_attention_2/truedivRealDiv+multi_head_self_attention_2/MatMul:output:0$multi_head_self_attention_2/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_2/truediv?
#multi_head_self_attention_2/SoftmaxSoftmax'multi_head_self_attention_2/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_2/Softmax?
$multi_head_self_attention_2/MatMul_1BatchMatMulV2-multi_head_self_attention_2/Softmax:softmax:0+multi_head_self_attention_2/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_2/MatMul_1?
,multi_head_self_attention_2/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_3/perm?
'multi_head_self_attention_2/transpose_3	Transpose-multi_head_self_attention_2/MatMul_1:output:05multi_head_self_attention_2/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_3?
-multi_head_self_attention_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_3/shape/1?
-multi_head_self_attention_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2/
-multi_head_self_attention_2/Reshape_3/shape/2?
+multi_head_self_attention_2/Reshape_3/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_3/shape/1:output:06multi_head_self_attention_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_3/shape?
%multi_head_self_attention_2/Reshape_3Reshape+multi_head_self_attention_2/transpose_3:y:04multi_head_self_attention_2/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2'
%multi_head_self_attention_2/Reshape_3?
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_21_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_21/Tensordot/axes?
3multi_head_self_attention_2/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_21/Tensordot/free?
4multi_head_self_attention_2/dense_21/Tensordot/ShapeShape.multi_head_self_attention_2/Reshape_3:output:0*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_21/Tensordot/Shape?
<multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_21/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_21/Tensordot/free:output:0Emulti_head_self_attention_2/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_21/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_21/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_21/Tensordot/Const?
3multi_head_self_attention_2/dense_21/Tensordot/ProdProd@multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_21/Tensordot/Prod?
6multi_head_self_attention_2/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_21/Tensordot/Const_1?
5multi_head_self_attention_2/dense_21/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_21/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_21/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_21/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_21/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_21/Tensordot/free:output:0<multi_head_self_attention_2/dense_21/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_21/Tensordot/concat?
4multi_head_self_attention_2/dense_21/Tensordot/stackPack<multi_head_self_attention_2/dense_21/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_21/Tensordot/stack?
8multi_head_self_attention_2/dense_21/Tensordot/transpose	Transpose.multi_head_self_attention_2/Reshape_3:output:0>multi_head_self_attention_2/dense_21/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2:
8multi_head_self_attention_2/dense_21/Tensordot/transpose?
6multi_head_self_attention_2/dense_21/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_21/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_21/Tensordot/Reshape?
5multi_head_self_attention_2/dense_21/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_21/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_21/Tensordot/MatMul?
6multi_head_self_attention_2/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_21/Tensordot/Const_2?
<multi_head_self_attention_2/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_21/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_21/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_21/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_21/Tensordot/concat_1?
.multi_head_self_attention_2/dense_21/TensordotReshape?multi_head_self_attention_2/dense_21/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_21/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_2/dense_21/Tensordot?
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_21/BiasAddBiasAdd7multi_head_self_attention_2/dense_21/Tensordot:output:0Cmulti_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2.
,multi_head_self_attention_2/dense_21/BiasAdd?
dropout_4/IdentityIdentity5multi_head_self_attention_2/dense_21/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_4/Identityn
addAddV2inputsdropout_4/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add?
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indices?
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2$
"layer_normalization_4/moments/mean?
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2,
*layer_normalization_4/moments/StopGradient?
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 21
/layer_normalization_4/moments/SquaredDifference?
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indices?
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2(
&layer_normalization_4/moments/variance?
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_4/batchnorm/add/y?
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2%
#layer_normalization_4/batchnorm/add?
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2'
%layer_normalization_4/batchnorm/Rsqrt?
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOp?
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_4/batchnorm/mul?
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/mul_1?
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/mul_2?
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOp?
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_4/batchnorm/sub?
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/add_1?
.sequential_2/dense_22/Tensordot/ReadVariableOpReadVariableOp7sequential_2_dense_22_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_2/dense_22/Tensordot/ReadVariableOp?
$sequential_2/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_2/dense_22/Tensordot/axes?
$sequential_2/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_2/dense_22/Tensordot/free?
%sequential_2/dense_22/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_2/dense_22/Tensordot/Shape?
-sequential_2/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_22/Tensordot/GatherV2/axis?
(sequential_2/dense_22/Tensordot/GatherV2GatherV2.sequential_2/dense_22/Tensordot/Shape:output:0-sequential_2/dense_22/Tensordot/free:output:06sequential_2/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_2/dense_22/Tensordot/GatherV2?
/sequential_2/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_2/dense_22/Tensordot/GatherV2_1/axis?
*sequential_2/dense_22/Tensordot/GatherV2_1GatherV2.sequential_2/dense_22/Tensordot/Shape:output:0-sequential_2/dense_22/Tensordot/axes:output:08sequential_2/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_2/dense_22/Tensordot/GatherV2_1?
%sequential_2/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_2/dense_22/Tensordot/Const?
$sequential_2/dense_22/Tensordot/ProdProd1sequential_2/dense_22/Tensordot/GatherV2:output:0.sequential_2/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_2/dense_22/Tensordot/Prod?
'sequential_2/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_22/Tensordot/Const_1?
&sequential_2/dense_22/Tensordot/Prod_1Prod3sequential_2/dense_22/Tensordot/GatherV2_1:output:00sequential_2/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_2/dense_22/Tensordot/Prod_1?
+sequential_2/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_2/dense_22/Tensordot/concat/axis?
&sequential_2/dense_22/Tensordot/concatConcatV2-sequential_2/dense_22/Tensordot/free:output:0-sequential_2/dense_22/Tensordot/axes:output:04sequential_2/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_2/dense_22/Tensordot/concat?
%sequential_2/dense_22/Tensordot/stackPack-sequential_2/dense_22/Tensordot/Prod:output:0/sequential_2/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_22/Tensordot/stack?
)sequential_2/dense_22/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0/sequential_2/dense_22/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_2/dense_22/Tensordot/transpose?
'sequential_2/dense_22/Tensordot/ReshapeReshape-sequential_2/dense_22/Tensordot/transpose:y:0.sequential_2/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_2/dense_22/Tensordot/Reshape?
&sequential_2/dense_22/Tensordot/MatMulMatMul0sequential_2/dense_22/Tensordot/Reshape:output:06sequential_2/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_2/dense_22/Tensordot/MatMul?
'sequential_2/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_22/Tensordot/Const_2?
-sequential_2/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_22/Tensordot/concat_1/axis?
(sequential_2/dense_22/Tensordot/concat_1ConcatV21sequential_2/dense_22/Tensordot/GatherV2:output:00sequential_2/dense_22/Tensordot/Const_2:output:06sequential_2/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_2/dense_22/Tensordot/concat_1?
sequential_2/dense_22/TensordotReshape0sequential_2/dense_22/Tensordot/MatMul:product:01sequential_2/dense_22/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_2/dense_22/Tensordot?
,sequential_2/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/dense_22/BiasAdd/ReadVariableOp?
sequential_2/dense_22/BiasAddBiasAdd(sequential_2/dense_22/Tensordot:output:04sequential_2/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_22/BiasAdd?
sequential_2/dense_22/ReluRelu&sequential_2/dense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_22/Relu?
.sequential_2/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_2_dense_23_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_2/dense_23/Tensordot/ReadVariableOp?
$sequential_2/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_2/dense_23/Tensordot/axes?
$sequential_2/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_2/dense_23/Tensordot/free?
%sequential_2/dense_23/Tensordot/ShapeShape(sequential_2/dense_22/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_2/dense_23/Tensordot/Shape?
-sequential_2/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_23/Tensordot/GatherV2/axis?
(sequential_2/dense_23/Tensordot/GatherV2GatherV2.sequential_2/dense_23/Tensordot/Shape:output:0-sequential_2/dense_23/Tensordot/free:output:06sequential_2/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_2/dense_23/Tensordot/GatherV2?
/sequential_2/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_2/dense_23/Tensordot/GatherV2_1/axis?
*sequential_2/dense_23/Tensordot/GatherV2_1GatherV2.sequential_2/dense_23/Tensordot/Shape:output:0-sequential_2/dense_23/Tensordot/axes:output:08sequential_2/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_2/dense_23/Tensordot/GatherV2_1?
%sequential_2/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_2/dense_23/Tensordot/Const?
$sequential_2/dense_23/Tensordot/ProdProd1sequential_2/dense_23/Tensordot/GatherV2:output:0.sequential_2/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_2/dense_23/Tensordot/Prod?
'sequential_2/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_23/Tensordot/Const_1?
&sequential_2/dense_23/Tensordot/Prod_1Prod3sequential_2/dense_23/Tensordot/GatherV2_1:output:00sequential_2/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_2/dense_23/Tensordot/Prod_1?
+sequential_2/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_2/dense_23/Tensordot/concat/axis?
&sequential_2/dense_23/Tensordot/concatConcatV2-sequential_2/dense_23/Tensordot/free:output:0-sequential_2/dense_23/Tensordot/axes:output:04sequential_2/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_2/dense_23/Tensordot/concat?
%sequential_2/dense_23/Tensordot/stackPack-sequential_2/dense_23/Tensordot/Prod:output:0/sequential_2/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_23/Tensordot/stack?
)sequential_2/dense_23/Tensordot/transpose	Transpose(sequential_2/dense_22/Relu:activations:0/sequential_2/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_2/dense_23/Tensordot/transpose?
'sequential_2/dense_23/Tensordot/ReshapeReshape-sequential_2/dense_23/Tensordot/transpose:y:0.sequential_2/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_2/dense_23/Tensordot/Reshape?
&sequential_2/dense_23/Tensordot/MatMulMatMul0sequential_2/dense_23/Tensordot/Reshape:output:06sequential_2/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_2/dense_23/Tensordot/MatMul?
'sequential_2/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_23/Tensordot/Const_2?
-sequential_2/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_23/Tensordot/concat_1/axis?
(sequential_2/dense_23/Tensordot/concat_1ConcatV21sequential_2/dense_23/Tensordot/GatherV2:output:00sequential_2/dense_23/Tensordot/Const_2:output:06sequential_2/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_2/dense_23/Tensordot/concat_1?
sequential_2/dense_23/TensordotReshape0sequential_2/dense_23/Tensordot/MatMul:product:01sequential_2/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_2/dense_23/Tensordot?
,sequential_2/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/dense_23/BiasAdd/ReadVariableOp?
sequential_2/dense_23/BiasAddBiasAdd(sequential_2/dense_23/Tensordot:output:04sequential_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_23/BiasAdd?
dropout_5/IdentityIdentity&sequential_2/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_5/Identity?
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_5/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add_1?
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indices?
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2$
"layer_normalization_5/moments/mean?
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2,
*layer_normalization_5/moments/StopGradient?
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 21
/layer_normalization_5/moments/SquaredDifference?
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indices?
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2(
&layer_normalization_5/moments/variance?
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_5/batchnorm/add/y?
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2%
#layer_normalization_5/batchnorm/add?
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2'
%layer_normalization_5/batchnorm/Rsqrt?
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOp?
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_5/batchnorm/mul?
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/mul_1?
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/mul_2?
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOp?
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_5/batchnorm/sub?
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/add_1?
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp<^multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp-^sequential_2/dense_22/BiasAdd/ReadVariableOp/^sequential_2/dense_22/Tensordot/ReadVariableOp-^sequential_2/dense_23/BiasAdd/ReadVariableOp/^sequential_2/dense_23/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2z
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp2\
,sequential_2/dense_22/BiasAdd/ReadVariableOp,sequential_2/dense_22/BiasAdd/ReadVariableOp2`
.sequential_2/dense_22/Tensordot/ReadVariableOp.sequential_2/dense_22/Tensordot/ReadVariableOp2\
,sequential_2/dense_23/BiasAdd/ReadVariableOp,sequential_2/dense_23/BiasAdd/ReadVariableOp2`
.sequential_2/dense_23/Tensordot/ReadVariableOp.sequential_2/dense_23/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?<
?
D__inference_model_2_layer_call_and_return_conditional_losses_1351156

inputs
inputs_18
&token_and_position_embedding_2_1350770:( 8
&token_and_position_embedding_2_1350772: -
transformer_block_2_1351020:  )
transformer_block_2_1351022: -
transformer_block_2_1351024:  )
transformer_block_2_1351026: -
transformer_block_2_1351028:  )
transformer_block_2_1351030: -
transformer_block_2_1351032:  )
transformer_block_2_1351034: )
transformer_block_2_1351036: )
transformer_block_2_1351038: -
transformer_block_2_1351040:  )
transformer_block_2_1351042: -
transformer_block_2_1351044:  )
transformer_block_2_1351046: )
transformer_block_2_1351048: )
transformer_block_2_1351050: $
aux_output_1351072:  
aux_output_1351074:"
dense_24_1351098:@
dense_24_1351100:@"
dense_25_1351115:@@
dense_25_1351117:@"
dense_26_1351132:@@
dense_26_1351134:@%
main_output_1351149:@!
main_output_1351151:
identity

identity_1??"aux_output/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?6token_and_position_embedding_2/StatefulPartitionedCall?+transformer_block_2/StatefulPartitionedCall?
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs&token_and_position_embedding_2_1350770&token_and_position_embedding_2_1350772*
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
GPU 2J 8? *d
f_R]
[__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_135076928
6token_and_position_embedding_2/StatefulPartitionedCall?
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_1351020transformer_block_2_1351022transformer_block_2_1351024transformer_block_2_1351026transformer_block_2_1351028transformer_block_2_1351030transformer_block_2_1351032transformer_block_2_1351034transformer_block_2_1351036transformer_block_2_1351038transformer_block_2_1351040transformer_block_2_1351042transformer_block_2_1351044transformer_block_2_1351046transformer_block_2_1351048transformer_block_2_1351050*
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
GPU 2J 8? *Y
fTRR
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_13510192-
+transformer_block_2/StatefulPartitionedCall?
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13510582,
*global_average_pooling1d_2/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0aux_output_1351072aux_output_1351074*
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
G__inference_aux_output_layer_call_and_return_conditional_losses_13510712$
"aux_output/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_13510842
concatenate_2/PartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_24_1351098dense_24_1351100*
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
GPU 2J 8? *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_13510972"
 dense_24/StatefulPartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_1351115dense_25_1351117*
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
GPU 2J 8? *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_13511142"
 dense_25/StatefulPartitionedCall?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_1351132dense_26_1351134*
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
GPU 2J 8? *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_13511312"
 dense_26/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0main_output_1351149main_output_1351151*
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
H__inference_main_output_layer_call_and_return_conditional_losses_13511482%
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
NoOpNoOp#^aux_output/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall$^main_output/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_1351577

inputsX
Fmulti_head_self_attention_2_dense_18_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_18_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_19_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_19_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_20_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_20_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_21_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_21_biasadd_readvariableop_resource: I
;layer_normalization_4_batchnorm_mul_readvariableop_resource: E
7layer_normalization_4_batchnorm_readvariableop_resource: I
7sequential_2_dense_22_tensordot_readvariableop_resource:  C
5sequential_2_dense_22_biasadd_readvariableop_resource: I
7sequential_2_dense_23_tensordot_readvariableop_resource:  C
5sequential_2_dense_23_biasadd_readvariableop_resource: I
;layer_normalization_5_batchnorm_mul_readvariableop_resource: E
7layer_normalization_5_batchnorm_readvariableop_resource: 
identity??.layer_normalization_4/batchnorm/ReadVariableOp?2layer_normalization_4/batchnorm/mul/ReadVariableOp?.layer_normalization_5/batchnorm/ReadVariableOp?2layer_normalization_5/batchnorm/mul/ReadVariableOp?;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?,sequential_2/dense_22/BiasAdd/ReadVariableOp?.sequential_2/dense_22/Tensordot/ReadVariableOp?,sequential_2/dense_23/BiasAdd/ReadVariableOp?.sequential_2/dense_23/Tensordot/ReadVariableOp|
!multi_head_self_attention_2/ShapeShapeinputs*
T0*
_output_shapes
:2#
!multi_head_self_attention_2/Shape?
/multi_head_self_attention_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention_2/strided_slice/stack?
1multi_head_self_attention_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_2/strided_slice/stack_1?
1multi_head_self_attention_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_2/strided_slice/stack_2?
)multi_head_self_attention_2/strided_sliceStridedSlice*multi_head_self_attention_2/Shape:output:08multi_head_self_attention_2/strided_slice/stack:output:0:multi_head_self_attention_2/strided_slice/stack_1:output:0:multi_head_self_attention_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention_2/strided_slice?
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_18_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_18/Tensordot/axes?
3multi_head_self_attention_2/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_18/Tensordot/free?
4multi_head_self_attention_2/dense_18/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_18/Tensordot/Shape?
<multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_18/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_18/Tensordot/free:output:0Emulti_head_self_attention_2/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_18/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_18/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_18/Tensordot/Const?
3multi_head_self_attention_2/dense_18/Tensordot/ProdProd@multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_18/Tensordot/Prod?
6multi_head_self_attention_2/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_18/Tensordot/Const_1?
5multi_head_self_attention_2/dense_18/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_18/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_18/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_18/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_18/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_18/Tensordot/free:output:0<multi_head_self_attention_2/dense_18/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_18/Tensordot/concat?
4multi_head_self_attention_2/dense_18/Tensordot/stackPack<multi_head_self_attention_2/dense_18/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_18/Tensordot/stack?
8multi_head_self_attention_2/dense_18/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_18/Tensordot/transpose?
6multi_head_self_attention_2/dense_18/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_18/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_18/Tensordot/Reshape?
5multi_head_self_attention_2/dense_18/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_18/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_18/Tensordot/MatMul?
6multi_head_self_attention_2/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_18/Tensordot/Const_2?
<multi_head_self_attention_2/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_18/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_18/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_18/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_18/Tensordot/concat_1?
.multi_head_self_attention_2/dense_18/TensordotReshape?multi_head_self_attention_2/dense_18/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_18/Tensordot?
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_18/BiasAddBiasAdd7multi_head_self_attention_2/dense_18/Tensordot:output:0Cmulti_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_18/BiasAdd?
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_19_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_19/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_19/Tensordot/axes?
3multi_head_self_attention_2/dense_19/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_19/Tensordot/free?
4multi_head_self_attention_2/dense_19/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_19/Tensordot/Shape?
<multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_19/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_19/Tensordot/free:output:0Emulti_head_self_attention_2/dense_19/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_19/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_19/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_19/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_19/Tensordot/Const?
3multi_head_self_attention_2/dense_19/Tensordot/ProdProd@multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_19/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_19/Tensordot/Prod?
6multi_head_self_attention_2/dense_19/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_19/Tensordot/Const_1?
5multi_head_self_attention_2/dense_19/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_19/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_19/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_19/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_19/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_19/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_19/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_19/Tensordot/free:output:0<multi_head_self_attention_2/dense_19/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_19/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_19/Tensordot/concat?
4multi_head_self_attention_2/dense_19/Tensordot/stackPack<multi_head_self_attention_2/dense_19/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_19/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_19/Tensordot/stack?
8multi_head_self_attention_2/dense_19/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_19/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_19/Tensordot/transpose?
6multi_head_self_attention_2/dense_19/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_19/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_19/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_19/Tensordot/Reshape?
5multi_head_self_attention_2/dense_19/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_19/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_19/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_19/Tensordot/MatMul?
6multi_head_self_attention_2/dense_19/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_19/Tensordot/Const_2?
<multi_head_self_attention_2/dense_19/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_19/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_19/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_19/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_19/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_19/Tensordot/concat_1?
.multi_head_self_attention_2/dense_19/TensordotReshape?multi_head_self_attention_2/dense_19/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_19/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_19/Tensordot?
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_19/BiasAddBiasAdd7multi_head_self_attention_2/dense_19/Tensordot:output:0Cmulti_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_19/BiasAdd?
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_20_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_20/Tensordot/axes?
3multi_head_self_attention_2/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_20/Tensordot/free?
4multi_head_self_attention_2/dense_20/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_20/Tensordot/Shape?
<multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_20/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_20/Tensordot/free:output:0Emulti_head_self_attention_2/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_20/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_20/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_20/Tensordot/Const?
3multi_head_self_attention_2/dense_20/Tensordot/ProdProd@multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_20/Tensordot/Prod?
6multi_head_self_attention_2/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_20/Tensordot/Const_1?
5multi_head_self_attention_2/dense_20/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_20/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_20/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_20/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_20/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_20/Tensordot/free:output:0<multi_head_self_attention_2/dense_20/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_20/Tensordot/concat?
4multi_head_self_attention_2/dense_20/Tensordot/stackPack<multi_head_self_attention_2/dense_20/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_20/Tensordot/stack?
8multi_head_self_attention_2/dense_20/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_20/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_20/Tensordot/transpose?
6multi_head_self_attention_2/dense_20/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_20/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_20/Tensordot/Reshape?
5multi_head_self_attention_2/dense_20/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_20/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_20/Tensordot/MatMul?
6multi_head_self_attention_2/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_20/Tensordot/Const_2?
<multi_head_self_attention_2/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_20/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_20/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_20/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_20/Tensordot/concat_1?
.multi_head_self_attention_2/dense_20/TensordotReshape?multi_head_self_attention_2/dense_20/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_20/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_20/Tensordot?
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_20/BiasAddBiasAdd7multi_head_self_attention_2/dense_20/Tensordot:output:0Cmulti_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_20/BiasAdd?
+multi_head_self_attention_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+multi_head_self_attention_2/Reshape/shape/1?
+multi_head_self_attention_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_2/Reshape/shape/2?
+multi_head_self_attention_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_2/Reshape/shape/3?
)multi_head_self_attention_2/Reshape/shapePack2multi_head_self_attention_2/strided_slice:output:04multi_head_self_attention_2/Reshape/shape/1:output:04multi_head_self_attention_2/Reshape/shape/2:output:04multi_head_self_attention_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention_2/Reshape/shape?
#multi_head_self_attention_2/ReshapeReshape5multi_head_self_attention_2/dense_18/BiasAdd:output:02multi_head_self_attention_2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2%
#multi_head_self_attention_2/Reshape?
*multi_head_self_attention_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention_2/transpose/perm?
%multi_head_self_attention_2/transpose	Transpose,multi_head_self_attention_2/Reshape:output:03multi_head_self_attention_2/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/transpose?
-multi_head_self_attention_2/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_1/shape/1?
-multi_head_self_attention_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_1/shape/2?
-multi_head_self_attention_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_1/shape/3?
+multi_head_self_attention_2/Reshape_1/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_1/shape/1:output:06multi_head_self_attention_2/Reshape_1/shape/2:output:06multi_head_self_attention_2/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_1/shape?
%multi_head_self_attention_2/Reshape_1Reshape5multi_head_self_attention_2/dense_19/BiasAdd:output:04multi_head_self_attention_2/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/Reshape_1?
,multi_head_self_attention_2/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_1/perm?
'multi_head_self_attention_2/transpose_1	Transpose.multi_head_self_attention_2/Reshape_1:output:05multi_head_self_attention_2/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_1?
-multi_head_self_attention_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_2/shape/1?
-multi_head_self_attention_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_2/shape/2?
-multi_head_self_attention_2/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_2/shape/3?
+multi_head_self_attention_2/Reshape_2/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_2/shape/1:output:06multi_head_self_attention_2/Reshape_2/shape/2:output:06multi_head_self_attention_2/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_2/shape?
%multi_head_self_attention_2/Reshape_2Reshape5multi_head_self_attention_2/dense_20/BiasAdd:output:04multi_head_self_attention_2/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/Reshape_2?
,multi_head_self_attention_2/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_2/perm?
'multi_head_self_attention_2/transpose_2	Transpose.multi_head_self_attention_2/Reshape_2:output:05multi_head_self_attention_2/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_2?
"multi_head_self_attention_2/MatMulBatchMatMulV2)multi_head_self_attention_2/transpose:y:0+multi_head_self_attention_2/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2$
"multi_head_self_attention_2/MatMul?
#multi_head_self_attention_2/Shape_1Shape+multi_head_self_attention_2/transpose_1:y:0*
T0*
_output_shapes
:2%
#multi_head_self_attention_2/Shape_1?
1multi_head_self_attention_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1multi_head_self_attention_2/strided_slice_1/stack?
3multi_head_self_attention_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention_2/strided_slice_1/stack_1?
3multi_head_self_attention_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/strided_slice_1/stack_2?
+multi_head_self_attention_2/strided_slice_1StridedSlice,multi_head_self_attention_2/Shape_1:output:0:multi_head_self_attention_2/strided_slice_1/stack:output:0<multi_head_self_attention_2/strided_slice_1/stack_1:output:0<multi_head_self_attention_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+multi_head_self_attention_2/strided_slice_1?
 multi_head_self_attention_2/CastCast4multi_head_self_attention_2/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 multi_head_self_attention_2/Cast?
 multi_head_self_attention_2/SqrtSqrt$multi_head_self_attention_2/Cast:y:0*
T0*
_output_shapes
: 2"
 multi_head_self_attention_2/Sqrt?
#multi_head_self_attention_2/truedivRealDiv+multi_head_self_attention_2/MatMul:output:0$multi_head_self_attention_2/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_2/truediv?
#multi_head_self_attention_2/SoftmaxSoftmax'multi_head_self_attention_2/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_2/Softmax?
$multi_head_self_attention_2/MatMul_1BatchMatMulV2-multi_head_self_attention_2/Softmax:softmax:0+multi_head_self_attention_2/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_2/MatMul_1?
,multi_head_self_attention_2/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_3/perm?
'multi_head_self_attention_2/transpose_3	Transpose-multi_head_self_attention_2/MatMul_1:output:05multi_head_self_attention_2/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_3?
-multi_head_self_attention_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_3/shape/1?
-multi_head_self_attention_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2/
-multi_head_self_attention_2/Reshape_3/shape/2?
+multi_head_self_attention_2/Reshape_3/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_3/shape/1:output:06multi_head_self_attention_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_3/shape?
%multi_head_self_attention_2/Reshape_3Reshape+multi_head_self_attention_2/transpose_3:y:04multi_head_self_attention_2/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2'
%multi_head_self_attention_2/Reshape_3?
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_21_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_21/Tensordot/axes?
3multi_head_self_attention_2/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_21/Tensordot/free?
4multi_head_self_attention_2/dense_21/Tensordot/ShapeShape.multi_head_self_attention_2/Reshape_3:output:0*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_21/Tensordot/Shape?
<multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_21/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_21/Tensordot/free:output:0Emulti_head_self_attention_2/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_21/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_21/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_21/Tensordot/Const?
3multi_head_self_attention_2/dense_21/Tensordot/ProdProd@multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_21/Tensordot/Prod?
6multi_head_self_attention_2/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_21/Tensordot/Const_1?
5multi_head_self_attention_2/dense_21/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_21/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_21/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_21/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_21/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_21/Tensordot/free:output:0<multi_head_self_attention_2/dense_21/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_21/Tensordot/concat?
4multi_head_self_attention_2/dense_21/Tensordot/stackPack<multi_head_self_attention_2/dense_21/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_21/Tensordot/stack?
8multi_head_self_attention_2/dense_21/Tensordot/transpose	Transpose.multi_head_self_attention_2/Reshape_3:output:0>multi_head_self_attention_2/dense_21/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2:
8multi_head_self_attention_2/dense_21/Tensordot/transpose?
6multi_head_self_attention_2/dense_21/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_21/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_21/Tensordot/Reshape?
5multi_head_self_attention_2/dense_21/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_21/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_21/Tensordot/MatMul?
6multi_head_self_attention_2/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_21/Tensordot/Const_2?
<multi_head_self_attention_2/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_21/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_21/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_21/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_21/Tensordot/concat_1?
.multi_head_self_attention_2/dense_21/TensordotReshape?multi_head_self_attention_2/dense_21/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_21/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_2/dense_21/Tensordot?
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_21/BiasAddBiasAdd7multi_head_self_attention_2/dense_21/Tensordot:output:0Cmulti_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2.
,multi_head_self_attention_2/dense_21/BiasAddw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMul5multi_head_self_attention_2/dense_21/BiasAdd:output:0 dropout_4/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeShape5multi_head_self_attention_2/dense_21/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_4/dropout/Mul_1n
addAddV2inputsdropout_4/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add?
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indices?
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2$
"layer_normalization_4/moments/mean?
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2,
*layer_normalization_4/moments/StopGradient?
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 21
/layer_normalization_4/moments/SquaredDifference?
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indices?
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2(
&layer_normalization_4/moments/variance?
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_4/batchnorm/add/y?
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2%
#layer_normalization_4/batchnorm/add?
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2'
%layer_normalization_4/batchnorm/Rsqrt?
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOp?
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_4/batchnorm/mul?
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/mul_1?
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/mul_2?
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOp?
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_4/batchnorm/sub?
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/add_1?
.sequential_2/dense_22/Tensordot/ReadVariableOpReadVariableOp7sequential_2_dense_22_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_2/dense_22/Tensordot/ReadVariableOp?
$sequential_2/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_2/dense_22/Tensordot/axes?
$sequential_2/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_2/dense_22/Tensordot/free?
%sequential_2/dense_22/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_2/dense_22/Tensordot/Shape?
-sequential_2/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_22/Tensordot/GatherV2/axis?
(sequential_2/dense_22/Tensordot/GatherV2GatherV2.sequential_2/dense_22/Tensordot/Shape:output:0-sequential_2/dense_22/Tensordot/free:output:06sequential_2/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_2/dense_22/Tensordot/GatherV2?
/sequential_2/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_2/dense_22/Tensordot/GatherV2_1/axis?
*sequential_2/dense_22/Tensordot/GatherV2_1GatherV2.sequential_2/dense_22/Tensordot/Shape:output:0-sequential_2/dense_22/Tensordot/axes:output:08sequential_2/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_2/dense_22/Tensordot/GatherV2_1?
%sequential_2/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_2/dense_22/Tensordot/Const?
$sequential_2/dense_22/Tensordot/ProdProd1sequential_2/dense_22/Tensordot/GatherV2:output:0.sequential_2/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_2/dense_22/Tensordot/Prod?
'sequential_2/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_22/Tensordot/Const_1?
&sequential_2/dense_22/Tensordot/Prod_1Prod3sequential_2/dense_22/Tensordot/GatherV2_1:output:00sequential_2/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_2/dense_22/Tensordot/Prod_1?
+sequential_2/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_2/dense_22/Tensordot/concat/axis?
&sequential_2/dense_22/Tensordot/concatConcatV2-sequential_2/dense_22/Tensordot/free:output:0-sequential_2/dense_22/Tensordot/axes:output:04sequential_2/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_2/dense_22/Tensordot/concat?
%sequential_2/dense_22/Tensordot/stackPack-sequential_2/dense_22/Tensordot/Prod:output:0/sequential_2/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_22/Tensordot/stack?
)sequential_2/dense_22/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0/sequential_2/dense_22/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_2/dense_22/Tensordot/transpose?
'sequential_2/dense_22/Tensordot/ReshapeReshape-sequential_2/dense_22/Tensordot/transpose:y:0.sequential_2/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_2/dense_22/Tensordot/Reshape?
&sequential_2/dense_22/Tensordot/MatMulMatMul0sequential_2/dense_22/Tensordot/Reshape:output:06sequential_2/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_2/dense_22/Tensordot/MatMul?
'sequential_2/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_22/Tensordot/Const_2?
-sequential_2/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_22/Tensordot/concat_1/axis?
(sequential_2/dense_22/Tensordot/concat_1ConcatV21sequential_2/dense_22/Tensordot/GatherV2:output:00sequential_2/dense_22/Tensordot/Const_2:output:06sequential_2/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_2/dense_22/Tensordot/concat_1?
sequential_2/dense_22/TensordotReshape0sequential_2/dense_22/Tensordot/MatMul:product:01sequential_2/dense_22/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_2/dense_22/Tensordot?
,sequential_2/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/dense_22/BiasAdd/ReadVariableOp?
sequential_2/dense_22/BiasAddBiasAdd(sequential_2/dense_22/Tensordot:output:04sequential_2/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_22/BiasAdd?
sequential_2/dense_22/ReluRelu&sequential_2/dense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_22/Relu?
.sequential_2/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_2_dense_23_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_2/dense_23/Tensordot/ReadVariableOp?
$sequential_2/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_2/dense_23/Tensordot/axes?
$sequential_2/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_2/dense_23/Tensordot/free?
%sequential_2/dense_23/Tensordot/ShapeShape(sequential_2/dense_22/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_2/dense_23/Tensordot/Shape?
-sequential_2/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_23/Tensordot/GatherV2/axis?
(sequential_2/dense_23/Tensordot/GatherV2GatherV2.sequential_2/dense_23/Tensordot/Shape:output:0-sequential_2/dense_23/Tensordot/free:output:06sequential_2/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_2/dense_23/Tensordot/GatherV2?
/sequential_2/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_2/dense_23/Tensordot/GatherV2_1/axis?
*sequential_2/dense_23/Tensordot/GatherV2_1GatherV2.sequential_2/dense_23/Tensordot/Shape:output:0-sequential_2/dense_23/Tensordot/axes:output:08sequential_2/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_2/dense_23/Tensordot/GatherV2_1?
%sequential_2/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_2/dense_23/Tensordot/Const?
$sequential_2/dense_23/Tensordot/ProdProd1sequential_2/dense_23/Tensordot/GatherV2:output:0.sequential_2/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_2/dense_23/Tensordot/Prod?
'sequential_2/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_23/Tensordot/Const_1?
&sequential_2/dense_23/Tensordot/Prod_1Prod3sequential_2/dense_23/Tensordot/GatherV2_1:output:00sequential_2/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_2/dense_23/Tensordot/Prod_1?
+sequential_2/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_2/dense_23/Tensordot/concat/axis?
&sequential_2/dense_23/Tensordot/concatConcatV2-sequential_2/dense_23/Tensordot/free:output:0-sequential_2/dense_23/Tensordot/axes:output:04sequential_2/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_2/dense_23/Tensordot/concat?
%sequential_2/dense_23/Tensordot/stackPack-sequential_2/dense_23/Tensordot/Prod:output:0/sequential_2/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_23/Tensordot/stack?
)sequential_2/dense_23/Tensordot/transpose	Transpose(sequential_2/dense_22/Relu:activations:0/sequential_2/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_2/dense_23/Tensordot/transpose?
'sequential_2/dense_23/Tensordot/ReshapeReshape-sequential_2/dense_23/Tensordot/transpose:y:0.sequential_2/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_2/dense_23/Tensordot/Reshape?
&sequential_2/dense_23/Tensordot/MatMulMatMul0sequential_2/dense_23/Tensordot/Reshape:output:06sequential_2/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_2/dense_23/Tensordot/MatMul?
'sequential_2/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_23/Tensordot/Const_2?
-sequential_2/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_23/Tensordot/concat_1/axis?
(sequential_2/dense_23/Tensordot/concat_1ConcatV21sequential_2/dense_23/Tensordot/GatherV2:output:00sequential_2/dense_23/Tensordot/Const_2:output:06sequential_2/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_2/dense_23/Tensordot/concat_1?
sequential_2/dense_23/TensordotReshape0sequential_2/dense_23/Tensordot/MatMul:product:01sequential_2/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_2/dense_23/Tensordot?
,sequential_2/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/dense_23/BiasAdd/ReadVariableOp?
sequential_2/dense_23/BiasAddBiasAdd(sequential_2/dense_23/Tensordot:output:04sequential_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_23/BiasAddw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMul&sequential_2/dense_23/BiasAdd:output:0 dropout_5/dropout/Const:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_5/dropout/Mul?
dropout_5/dropout/ShapeShape&sequential_2/dense_23/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????( *
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????( 2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????( 2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????( 2
dropout_5/dropout/Mul_1?
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_5/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add_1?
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indices?
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2$
"layer_normalization_5/moments/mean?
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2,
*layer_normalization_5/moments/StopGradient?
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 21
/layer_normalization_5/moments/SquaredDifference?
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indices?
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2(
&layer_normalization_5/moments/variance?
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_5/batchnorm/add/y?
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2%
#layer_normalization_5/batchnorm/add?
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2'
%layer_normalization_5/batchnorm/Rsqrt?
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOp?
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_5/batchnorm/mul?
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/mul_1?
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/mul_2?
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOp?
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_5/batchnorm/sub?
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/add_1?
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp<^multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp-^sequential_2/dense_22/BiasAdd/ReadVariableOp/^sequential_2/dense_22/Tensordot/ReadVariableOp-^sequential_2/dense_23/BiasAdd/ReadVariableOp/^sequential_2/dense_23/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2z
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp2\
,sequential_2/dense_22/BiasAdd/ReadVariableOp,sequential_2/dense_22/BiasAdd/ReadVariableOp2`
.sequential_2/dense_22/Tensordot/ReadVariableOp.sequential_2/dense_22/Tensordot/ReadVariableOp2\
,sequential_2/dense_23/BiasAdd/ReadVariableOp,sequential_2/dense_23/BiasAdd/ReadVariableOp2`
.sequential_2/dense_23/Tensordot/ReadVariableOp.sequential_2/dense_23/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
)__inference_model_2_layer_call_fn_1352163
inputs_0
inputs_1
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

unknown_19:@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_13511562
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
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
inputs/1
?
t
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1351084

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
D__inference_model_2_layer_call_and_return_conditional_losses_1351760

inputs
inputs_18
&token_and_position_embedding_2_1351693:( 8
&token_and_position_embedding_2_1351695: -
transformer_block_2_1351698:  )
transformer_block_2_1351700: -
transformer_block_2_1351702:  )
transformer_block_2_1351704: -
transformer_block_2_1351706:  )
transformer_block_2_1351708: -
transformer_block_2_1351710:  )
transformer_block_2_1351712: )
transformer_block_2_1351714: )
transformer_block_2_1351716: -
transformer_block_2_1351718:  )
transformer_block_2_1351720: -
transformer_block_2_1351722:  )
transformer_block_2_1351724: )
transformer_block_2_1351726: )
transformer_block_2_1351728: $
aux_output_1351732:  
aux_output_1351734:"
dense_24_1351738:@
dense_24_1351740:@"
dense_25_1351743:@@
dense_25_1351745:@"
dense_26_1351748:@@
dense_26_1351750:@%
main_output_1351753:@!
main_output_1351755:
identity

identity_1??"aux_output/StatefulPartitionedCall? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall? dense_26/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?6token_and_position_embedding_2/StatefulPartitionedCall?+transformer_block_2/StatefulPartitionedCall?
6token_and_position_embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs&token_and_position_embedding_2_1351693&token_and_position_embedding_2_1351695*
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
GPU 2J 8? *d
f_R]
[__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_135076928
6token_and_position_embedding_2/StatefulPartitionedCall?
+transformer_block_2/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_2/StatefulPartitionedCall:output:0transformer_block_2_1351698transformer_block_2_1351700transformer_block_2_1351702transformer_block_2_1351704transformer_block_2_1351706transformer_block_2_1351708transformer_block_2_1351710transformer_block_2_1351712transformer_block_2_1351714transformer_block_2_1351716transformer_block_2_1351718transformer_block_2_1351720transformer_block_2_1351722transformer_block_2_1351724transformer_block_2_1351726transformer_block_2_1351728*
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
GPU 2J 8? *Y
fTRR
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_13515772-
+transformer_block_2/StatefulPartitionedCall?
*global_average_pooling1d_2/PartitionedCallPartitionedCall4transformer_block_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *`
f[RY
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_13510582,
*global_average_pooling1d_2/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_2/PartitionedCall:output:0aux_output_1351732aux_output_1351734*
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
G__inference_aux_output_layer_call_and_return_conditional_losses_13510712$
"aux_output/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_13510842
concatenate_2/PartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_24_1351738dense_24_1351740*
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
GPU 2J 8? *N
fIRG
E__inference_dense_24_layer_call_and_return_conditional_losses_13510972"
 dense_24/StatefulPartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_1351743dense_25_1351745*
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
GPU 2J 8? *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_13511142"
 dense_25/StatefulPartitionedCall?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_25/StatefulPartitionedCall:output:0dense_26_1351748dense_26_1351750*
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
GPU 2J 8? *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_13511312"
 dense_26/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0main_output_1351753main_output_1351755*
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
H__inference_main_output_layer_call_and_return_conditional_losses_13511482%
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
NoOpNoOp#^aux_output/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall$^main_output/StatefulPartitionedCall7^token_and_position_embedding_2/StatefulPartitionedCall,^transformer_block_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2p
6token_and_position_embedding_2/StatefulPartitionedCall6token_and_position_embedding_2/StatefulPartitionedCall2Z
+transformer_block_2/StatefulPartitionedCall+transformer_block_2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
[__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_1352884
x6
$embedding_5_embedding_lookup_1352871:( 6
$embedding_4_embedding_lookup_1352877: 
identity??embedding_4/embedding_lookup?embedding_5/embedding_lookup?
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
embedding_5/embedding_lookupResourceGather$embedding_5_embedding_lookup_1352871range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_5/embedding_lookup/1352871*'
_output_shapes
:????????? *
dtype02
embedding_5/embedding_lookup?
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_5/embedding_lookup/1352871*'
_output_shapes
:????????? 2'
%embedding_5/embedding_lookup/Identity?
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_5/embedding_lookup/Identity_1p
embedding_4/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:?????????(2
embedding_4/Cast?
embedding_4/embedding_lookupResourceGather$embedding_4_embedding_lookup_1352877embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_4/embedding_lookup/1352877*+
_output_shapes
:?????????( *
dtype02
embedding_4/embedding_lookup?
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_4/embedding_lookup/1352877*+
_output_shapes
:?????????( 2'
%embedding_4/embedding_lookup/Identity?
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2)
'embedding_4/embedding_lookup/Identity_1?
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2
addf
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp^embedding_4/embedding_lookup^embedding_5/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:J F
'
_output_shapes
:?????????(

_user_specified_namex
?
?
*__inference_dense_22_layer_call_fn_1353744

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
GPU 2J 8? *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_13505572
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
?
?
H__inference_main_output_layer_call_and_return_conditional_losses_1351148

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
?
?
5__inference_transformer_block_2_layer_call_fn_1352921

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
GPU 2J 8? *Y
fTRR
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_13510192
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
? 
?
E__inference_dense_23_layer_call_and_return_conditional_losses_1353814

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
?#
"__inference__wrapped_model_1350519
input_3
	aux_input]
Kmodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_1350226:( ]
Kmodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_1350232: t
bmodel_2_transformer_block_2_multi_head_self_attention_2_dense_18_tensordot_readvariableop_resource:  n
`model_2_transformer_block_2_multi_head_self_attention_2_dense_18_biasadd_readvariableop_resource: t
bmodel_2_transformer_block_2_multi_head_self_attention_2_dense_19_tensordot_readvariableop_resource:  n
`model_2_transformer_block_2_multi_head_self_attention_2_dense_19_biasadd_readvariableop_resource: t
bmodel_2_transformer_block_2_multi_head_self_attention_2_dense_20_tensordot_readvariableop_resource:  n
`model_2_transformer_block_2_multi_head_self_attention_2_dense_20_biasadd_readvariableop_resource: t
bmodel_2_transformer_block_2_multi_head_self_attention_2_dense_21_tensordot_readvariableop_resource:  n
`model_2_transformer_block_2_multi_head_self_attention_2_dense_21_biasadd_readvariableop_resource: e
Wmodel_2_transformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource: a
Smodel_2_transformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource: e
Smodel_2_transformer_block_2_sequential_2_dense_22_tensordot_readvariableop_resource:  _
Qmodel_2_transformer_block_2_sequential_2_dense_22_biasadd_readvariableop_resource: e
Smodel_2_transformer_block_2_sequential_2_dense_23_tensordot_readvariableop_resource:  _
Qmodel_2_transformer_block_2_sequential_2_dense_23_biasadd_readvariableop_resource: e
Wmodel_2_transformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource: a
Smodel_2_transformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource: C
1model_2_aux_output_matmul_readvariableop_resource: @
2model_2_aux_output_biasadd_readvariableop_resource:A
/model_2_dense_24_matmul_readvariableop_resource:@>
0model_2_dense_24_biasadd_readvariableop_resource:@A
/model_2_dense_25_matmul_readvariableop_resource:@@>
0model_2_dense_25_biasadd_readvariableop_resource:@A
/model_2_dense_26_matmul_readvariableop_resource:@@>
0model_2_dense_26_biasadd_readvariableop_resource:@D
2model_2_main_output_matmul_readvariableop_resource:@A
3model_2_main_output_biasadd_readvariableop_resource:
identity

identity_1??)model_2/aux_output/BiasAdd/ReadVariableOp?(model_2/aux_output/MatMul/ReadVariableOp?'model_2/dense_24/BiasAdd/ReadVariableOp?&model_2/dense_24/MatMul/ReadVariableOp?'model_2/dense_25/BiasAdd/ReadVariableOp?&model_2/dense_25/MatMul/ReadVariableOp?'model_2/dense_26/BiasAdd/ReadVariableOp?&model_2/dense_26/MatMul/ReadVariableOp?*model_2/main_output/BiasAdd/ReadVariableOp?)model_2/main_output/MatMul/ReadVariableOp?Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup?Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup?Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp?Nmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp?Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp?Nmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp?Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?Hmodel_2/transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp?Jmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp?Hmodel_2/transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp?Jmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp?
,model_2/token_and_position_embedding_2/ShapeShapeinput_3*
T0*
_output_shapes
:2.
,model_2/token_and_position_embedding_2/Shape?
:model_2/token_and_position_embedding_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2<
:model_2/token_and_position_embedding_2/strided_slice/stack?
<model_2/token_and_position_embedding_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_2/token_and_position_embedding_2/strided_slice/stack_1?
<model_2/token_and_position_embedding_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_2/token_and_position_embedding_2/strided_slice/stack_2?
4model_2/token_and_position_embedding_2/strided_sliceStridedSlice5model_2/token_and_position_embedding_2/Shape:output:0Cmodel_2/token_and_position_embedding_2/strided_slice/stack:output:0Emodel_2/token_and_position_embedding_2/strided_slice/stack_1:output:0Emodel_2/token_and_position_embedding_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_2/token_and_position_embedding_2/strided_slice?
2model_2/token_and_position_embedding_2/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_2/token_and_position_embedding_2/range/start?
2model_2/token_and_position_embedding_2/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_2/token_and_position_embedding_2/range/delta?
,model_2/token_and_position_embedding_2/rangeRange;model_2/token_and_position_embedding_2/range/start:output:0=model_2/token_and_position_embedding_2/strided_slice:output:0;model_2/token_and_position_embedding_2/range/delta:output:0*#
_output_shapes
:?????????2.
,model_2/token_and_position_embedding_2/range?
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupResourceGatherKmodel_2_token_and_position_embedding_2_embedding_5_embedding_lookup_13502265model_2/token_and_position_embedding_2/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*^
_classT
RPloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/1350226*'
_output_shapes
:????????? *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup?
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*^
_classT
RPloc:@model_2/token_and_position_embedding_2/embedding_5/embedding_lookup/1350226*'
_output_shapes
:????????? 2N
Lmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity?
Nmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2P
Nmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1?
7model_2/token_and_position_embedding_2/embedding_4/CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:?????????(29
7model_2/token_and_position_embedding_2/embedding_4/Cast?
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupResourceGatherKmodel_2_token_and_position_embedding_2_embedding_4_embedding_lookup_1350232;model_2/token_and_position_embedding_2/embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*^
_classT
RPloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/1350232*+
_output_shapes
:?????????( *
dtype02E
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup?
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/IdentityIdentityLmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*^
_classT
RPloc:@model_2/token_and_position_embedding_2/embedding_4/embedding_lookup/1350232*+
_output_shapes
:?????????( 2N
Lmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity?
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1IdentityUmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2P
Nmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1?
*model_2/token_and_position_embedding_2/addAddV2Wmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup/Identity_1:output:0Wmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2,
*model_2/token_and_position_embedding_2/add?
=model_2/transformer_block_2/multi_head_self_attention_2/ShapeShape.model_2/token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:2?
=model_2/transformer_block_2/multi_head_self_attention_2/Shape?
Kmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Kmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice/stack?
Mmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Mmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice/stack_1?
Mmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Mmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice/stack_2?
Emodel_2/transformer_block_2/multi_head_self_attention_2/strided_sliceStridedSliceFmodel_2/transformer_block_2/multi_head_self_attention_2/Shape:output:0Tmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice/stack:output:0Vmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice/stack_1:output:0Vmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2G
Emodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice?
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpReadVariableOpbmodel_2_transformer_block_2_multi_head_self_attention_2_dense_18_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02[
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axes?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/free?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ShapeShape.model_2/token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Shape?
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axis?
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2GatherV2Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/free:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2?
Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis?
Umodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1GatherV2Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axes:output:0cmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Umodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ProdProd\model_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_1?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod_1Prod^model_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1:output:0[model_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod_1?
Vmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat/axis?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concatConcatV2Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/free:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/axes:output:0_model_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/stackPackXmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod:output:0Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/stack?
Tmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/transpose	Transpose.model_2/token_and_position_embedding_2/add:z:0Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2V
Tmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/transpose?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReshapeReshapeXmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/transpose:y:0Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Reshape?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/MatMulMatMul[model_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Reshape:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/MatMul?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_2?
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1/axis?
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1ConcatV2\model_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0[model_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/Const_2:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1?
Jmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/TensordotReshape[model_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/MatMul:product:0\model_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2L
Jmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot?
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpReadVariableOp`model_2_transformer_block_2_multi_head_self_attention_2_dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?
Hmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/BiasAddBiasAddSmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot:output:0_model_2/transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2J
Hmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd?
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpReadVariableOpbmodel_2_transformer_block_2_multi_head_self_attention_2_dense_19_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02[
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axes?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/free?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ShapeShape.model_2/token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Shape?
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axis?
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2GatherV2Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/free:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2?
Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis?
Umodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1GatherV2Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axes:output:0cmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Umodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ProdProd\model_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_1?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod_1Prod^model_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1:output:0[model_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod_1?
Vmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat/axis?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concatConcatV2Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/free:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/axes:output:0_model_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/stackPackXmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod:output:0Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/stack?
Tmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/transpose	Transpose.model_2/token_and_position_embedding_2/add:z:0Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2V
Tmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/transpose?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReshapeReshapeXmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/transpose:y:0Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Reshape?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/MatMulMatMul[model_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Reshape:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/MatMul?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_2?
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1/axis?
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1ConcatV2\model_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0[model_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/Const_2:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1?
Jmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/TensordotReshape[model_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/MatMul:product:0\model_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2L
Jmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot?
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpReadVariableOp`model_2_transformer_block_2_multi_head_self_attention_2_dense_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?
Hmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/BiasAddBiasAddSmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot:output:0_model_2/transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2J
Hmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd?
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpReadVariableOpbmodel_2_transformer_block_2_multi_head_self_attention_2_dense_20_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02[
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axes?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/free?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ShapeShape.model_2/token_and_position_embedding_2/add:z:0*
T0*
_output_shapes
:2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Shape?
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axis?
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2GatherV2Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/free:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2?
Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis?
Umodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1GatherV2Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axes:output:0cmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Umodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ProdProd\model_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_1?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod_1Prod^model_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1:output:0[model_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod_1?
Vmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat/axis?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concatConcatV2Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/free:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/axes:output:0_model_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/stackPackXmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod:output:0Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/stack?
Tmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/transpose	Transpose.model_2/token_and_position_embedding_2/add:z:0Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2V
Tmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/transpose?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReshapeReshapeXmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/transpose:y:0Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Reshape?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/MatMulMatMul[model_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Reshape:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/MatMul?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_2?
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1/axis?
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1ConcatV2\model_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0[model_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/Const_2:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1?
Jmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/TensordotReshape[model_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/MatMul:product:0\model_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2L
Jmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot?
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpReadVariableOp`model_2_transformer_block_2_multi_head_self_attention_2_dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?
Hmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/BiasAddBiasAddSmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot:output:0_model_2/transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2J
Hmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd?
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2I
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape/1?
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2I
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape/2?
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2I
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape/3?
Emodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shapePackNmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice:output:0Pmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape/1:output:0Pmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape/2:output:0Pmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2G
Emodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape?
?model_2/transformer_block_2/multi_head_self_attention_2/ReshapeReshapeQmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd:output:0Nmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2A
?model_2/transformer_block_2/multi_head_self_attention_2/Reshape?
Fmodel_2/transformer_block_2/multi_head_self_attention_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2H
Fmodel_2/transformer_block_2/multi_head_self_attention_2/transpose/perm?
Amodel_2/transformer_block_2/multi_head_self_attention_2/transpose	TransposeHmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape:output:0Omodel_2/transformer_block_2/multi_head_self_attention_2/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2C
Amodel_2/transformer_block_2/multi_head_self_attention_2/transpose?
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2K
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape/1?
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2K
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape/2?
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2K
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape/3?
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shapePackNmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice:output:0Rmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape/1:output:0Rmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape/2:output:0Rmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2I
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape?
Amodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1ReshapeQmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd:output:0Pmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2C
Amodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1?
Hmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2J
Hmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_1/perm?
Cmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_1	TransposeJmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_1:output:0Qmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2E
Cmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_1?
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2K
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape/1?
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2K
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape/2?
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2K
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape/3?
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shapePackNmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice:output:0Rmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape/1:output:0Rmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape/2:output:0Rmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2I
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape?
Amodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2ReshapeQmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd:output:0Pmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2C
Amodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2?
Hmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2J
Hmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_2/perm?
Cmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_2	TransposeJmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_2:output:0Qmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2E
Cmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_2?
>model_2/transformer_block_2/multi_head_self_attention_2/MatMulBatchMatMulV2Emodel_2/transformer_block_2/multi_head_self_attention_2/transpose:y:0Gmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2@
>model_2/transformer_block_2/multi_head_self_attention_2/MatMul?
?model_2/transformer_block_2/multi_head_self_attention_2/Shape_1ShapeGmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_1:y:0*
T0*
_output_shapes
:2A
?model_2/transformer_block_2/multi_head_self_attention_2/Shape_1?
Mmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2O
Mmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1/stack?
Omodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_1?
Omodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_2?
Gmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1StridedSliceHmodel_2/transformer_block_2/multi_head_self_attention_2/Shape_1:output:0Vmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1/stack:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_1:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1?
<model_2/transformer_block_2/multi_head_self_attention_2/CastCastPmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2>
<model_2/transformer_block_2/multi_head_self_attention_2/Cast?
<model_2/transformer_block_2/multi_head_self_attention_2/SqrtSqrt@model_2/transformer_block_2/multi_head_self_attention_2/Cast:y:0*
T0*
_output_shapes
: 2>
<model_2/transformer_block_2/multi_head_self_attention_2/Sqrt?
?model_2/transformer_block_2/multi_head_self_attention_2/truedivRealDivGmodel_2/transformer_block_2/multi_head_self_attention_2/MatMul:output:0@model_2/transformer_block_2/multi_head_self_attention_2/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2A
?model_2/transformer_block_2/multi_head_self_attention_2/truediv?
?model_2/transformer_block_2/multi_head_self_attention_2/SoftmaxSoftmaxCmodel_2/transformer_block_2/multi_head_self_attention_2/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2A
?model_2/transformer_block_2/multi_head_self_attention_2/Softmax?
@model_2/transformer_block_2/multi_head_self_attention_2/MatMul_1BatchMatMulV2Imodel_2/transformer_block_2/multi_head_self_attention_2/Softmax:softmax:0Gmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2B
@model_2/transformer_block_2/multi_head_self_attention_2/MatMul_1?
Hmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2J
Hmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_3/perm?
Cmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_3	TransposeImodel_2/transformer_block_2/multi_head_self_attention_2/MatMul_1:output:0Qmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2E
Cmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_3?
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2K
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3/shape/1?
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3/shape/2?
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3/shapePackNmodel_2/transformer_block_2/multi_head_self_attention_2/strided_slice:output:0Rmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3/shape/1:output:0Rmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2I
Gmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3/shape?
Amodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3ReshapeGmodel_2/transformer_block_2/multi_head_self_attention_2/transpose_3:y:0Pmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2C
Amodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3?
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpReadVariableOpbmodel_2_transformer_block_2_multi_head_self_attention_2_dense_21_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02[
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axes?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/free?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ShapeShapeJmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3:output:0*
T0*
_output_shapes
:2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Shape?
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axis?
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2GatherV2Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/free:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2?
Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis?
Umodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1GatherV2Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axes:output:0cmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Umodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const?
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ProdProd\model_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Omodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_1?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod_1Prod^model_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1:output:0[model_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod_1?
Vmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat/axis?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concatConcatV2Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/free:output:0Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/axes:output:0_model_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat?
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/stackPackXmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod:output:0Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/stack?
Tmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/transpose	TransposeJmodel_2/transformer_block_2/multi_head_self_attention_2/Reshape_3:output:0Zmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2V
Tmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/transpose?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReshapeReshapeXmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/transpose:y:0Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Reshape?
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/MatMulMatMul[model_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Reshape:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2S
Qmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/MatMul?
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_2?
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1/axis?
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1ConcatV2\model_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0[model_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/Const_2:output:0amodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Smodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1?
Jmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/TensordotReshape[model_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/MatMul:product:0\model_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 2L
Jmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot?
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpReadVariableOp`model_2_transformer_block_2_multi_head_self_attention_2_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?
Hmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/BiasAddBiasAddSmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot:output:0_model_2/transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2J
Hmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd?
.model_2/transformer_block_2/dropout_4/IdentityIdentityQmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 20
.model_2/transformer_block_2/dropout_4/Identity?
model_2/transformer_block_2/addAddV2.model_2/token_and_position_embedding_2/add:z:07model_2/transformer_block_2/dropout_4/Identity:output:0*
T0*+
_output_shapes
:?????????( 2!
model_2/transformer_block_2/add?
Pmodel_2/transformer_block_2/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_2/transformer_block_2/layer_normalization_4/moments/mean/reduction_indices?
>model_2/transformer_block_2/layer_normalization_4/moments/meanMean#model_2/transformer_block_2/add:z:0Ymodel_2/transformer_block_2/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2@
>model_2/transformer_block_2/layer_normalization_4/moments/mean?
Fmodel_2/transformer_block_2/layer_normalization_4/moments/StopGradientStopGradientGmodel_2/transformer_block_2/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2H
Fmodel_2/transformer_block_2/layer_normalization_4/moments/StopGradient?
Kmodel_2/transformer_block_2/layer_normalization_4/moments/SquaredDifferenceSquaredDifference#model_2/transformer_block_2/add:z:0Omodel_2/transformer_block_2/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2M
Kmodel_2/transformer_block_2/layer_normalization_4/moments/SquaredDifference?
Tmodel_2/transformer_block_2/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2V
Tmodel_2/transformer_block_2/layer_normalization_4/moments/variance/reduction_indices?
Bmodel_2/transformer_block_2/layer_normalization_4/moments/varianceMeanOmodel_2/transformer_block_2/layer_normalization_4/moments/SquaredDifference:z:0]model_2/transformer_block_2/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2D
Bmodel_2/transformer_block_2/layer_normalization_4/moments/variance?
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add/y?
?model_2/transformer_block_2/layer_normalization_4/batchnorm/addAddV2Kmodel_2/transformer_block_2/layer_normalization_4/moments/variance:output:0Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2A
?model_2/transformer_block_2/layer_normalization_4/batchnorm/add?
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/RsqrtRsqrtCmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/Rsqrt?
Nmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_2_transformer_block_2_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02P
Nmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp?
?model_2/transformer_block_2/layer_normalization_4/batchnorm/mulMulEmodel_2/transformer_block_2/layer_normalization_4/batchnorm/Rsqrt:y:0Vmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2A
?model_2/transformer_block_2/layer_normalization_4/batchnorm/mul?
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_1Mul#model_2/transformer_block_2/add:z:0Cmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_1?
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_2MulGmodel_2/transformer_block_2/layer_normalization_4/moments/mean:output:0Cmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_2?
Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpSmodel_2_transformer_block_2_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp?
?model_2/transformer_block_2/layer_normalization_4/batchnorm/subSubRmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp:value:0Emodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2A
?model_2/transformer_block_2/layer_normalization_4/batchnorm/sub?
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1AddV2Emodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul_1:z:0Cmodel_2/transformer_block_2/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2C
Amodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1?
Jmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOpReadVariableOpSmodel_2_transformer_block_2_sequential_2_dense_22_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02L
Jmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp?
@model_2/transformer_block_2/sequential_2/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_2/transformer_block_2/sequential_2/dense_22/Tensordot/axes?
@model_2/transformer_block_2/sequential_2/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_2/transformer_block_2/sequential_2/dense_22/Tensordot/free?
Amodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/ShapeShapeEmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Shape?
Imodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2/axis?
Dmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2GatherV2Jmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Shape:output:0Imodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/free:output:0Rmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2?
Kmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1/axis?
Fmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1GatherV2Jmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Shape:output:0Imodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/axes:output:0Tmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1?
Amodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Const?
@model_2/transformer_block_2/sequential_2/dense_22/Tensordot/ProdProdMmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2:output:0Jmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_2/transformer_block_2/sequential_2/dense_22/Tensordot/Prod?
Cmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Const_1?
Bmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Prod_1ProdOmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2_1:output:0Lmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Prod_1?
Gmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat/axis?
Bmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concatConcatV2Imodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/free:output:0Imodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/axes:output:0Pmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat?
Amodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/stackPackImodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Prod:output:0Kmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/stack?
Emodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/transpose	TransposeEmodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1:z:0Kmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2G
Emodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/transpose?
Cmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/ReshapeReshapeImodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/transpose:y:0Jmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2E
Cmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Reshape?
Bmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/MatMulMatMulLmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Reshape:output:0Rmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2D
Bmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/MatMul?
Cmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Const_2?
Imodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat_1/axis?
Dmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat_1ConcatV2Mmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/GatherV2:output:0Lmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/Const_2:output:0Rmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat_1?
;model_2/transformer_block_2/sequential_2/dense_22/TensordotReshapeLmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/MatMul:product:0Mmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2=
;model_2/transformer_block_2/sequential_2/dense_22/Tensordot?
Hmodel_2/transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_transformer_block_2_sequential_2_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_2/transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp?
9model_2/transformer_block_2/sequential_2/dense_22/BiasAddBiasAddDmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot:output:0Pmodel_2/transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2;
9model_2/transformer_block_2/sequential_2/dense_22/BiasAdd?
6model_2/transformer_block_2/sequential_2/dense_22/ReluReluBmodel_2/transformer_block_2/sequential_2/dense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 28
6model_2/transformer_block_2/sequential_2/dense_22/Relu?
Jmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOpReadVariableOpSmodel_2_transformer_block_2_sequential_2_dense_23_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02L
Jmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp?
@model_2/transformer_block_2/sequential_2/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_2/transformer_block_2/sequential_2/dense_23/Tensordot/axes?
@model_2/transformer_block_2/sequential_2/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_2/transformer_block_2/sequential_2/dense_23/Tensordot/free?
Amodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/ShapeShapeDmodel_2/transformer_block_2/sequential_2/dense_22/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Shape?
Imodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2/axis?
Dmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2GatherV2Jmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Shape:output:0Imodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/free:output:0Rmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2?
Kmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1/axis?
Fmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1GatherV2Jmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Shape:output:0Imodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/axes:output:0Tmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1?
Amodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Const?
@model_2/transformer_block_2/sequential_2/dense_23/Tensordot/ProdProdMmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2:output:0Jmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_2/transformer_block_2/sequential_2/dense_23/Tensordot/Prod?
Cmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Const_1?
Bmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Prod_1ProdOmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2_1:output:0Lmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Prod_1?
Gmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat/axis?
Bmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concatConcatV2Imodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/free:output:0Imodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/axes:output:0Pmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat?
Amodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/stackPackImodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Prod:output:0Kmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/stack?
Emodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/transpose	TransposeDmodel_2/transformer_block_2/sequential_2/dense_22/Relu:activations:0Kmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2G
Emodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/transpose?
Cmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/ReshapeReshapeImodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/transpose:y:0Jmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2E
Cmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Reshape?
Bmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/MatMulMatMulLmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Reshape:output:0Rmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2D
Bmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/MatMul?
Cmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Const_2?
Imodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat_1/axis?
Dmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat_1ConcatV2Mmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/GatherV2:output:0Lmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/Const_2:output:0Rmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat_1?
;model_2/transformer_block_2/sequential_2/dense_23/TensordotReshapeLmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/MatMul:product:0Mmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2=
;model_2/transformer_block_2/sequential_2/dense_23/Tensordot?
Hmodel_2/transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOpReadVariableOpQmodel_2_transformer_block_2_sequential_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_2/transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp?
9model_2/transformer_block_2/sequential_2/dense_23/BiasAddBiasAddDmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot:output:0Pmodel_2/transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2;
9model_2/transformer_block_2/sequential_2/dense_23/BiasAdd?
.model_2/transformer_block_2/dropout_5/IdentityIdentityBmodel_2/transformer_block_2/sequential_2/dense_23/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 20
.model_2/transformer_block_2/dropout_5/Identity?
!model_2/transformer_block_2/add_1AddV2Emodel_2/transformer_block_2/layer_normalization_4/batchnorm/add_1:z:07model_2/transformer_block_2/dropout_5/Identity:output:0*
T0*+
_output_shapes
:?????????( 2#
!model_2/transformer_block_2/add_1?
Pmodel_2/transformer_block_2/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
Pmodel_2/transformer_block_2/layer_normalization_5/moments/mean/reduction_indices?
>model_2/transformer_block_2/layer_normalization_5/moments/meanMean%model_2/transformer_block_2/add_1:z:0Ymodel_2/transformer_block_2/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2@
>model_2/transformer_block_2/layer_normalization_5/moments/mean?
Fmodel_2/transformer_block_2/layer_normalization_5/moments/StopGradientStopGradientGmodel_2/transformer_block_2/layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2H
Fmodel_2/transformer_block_2/layer_normalization_5/moments/StopGradient?
Kmodel_2/transformer_block_2/layer_normalization_5/moments/SquaredDifferenceSquaredDifference%model_2/transformer_block_2/add_1:z:0Omodel_2/transformer_block_2/layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2M
Kmodel_2/transformer_block_2/layer_normalization_5/moments/SquaredDifference?
Tmodel_2/transformer_block_2/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2V
Tmodel_2/transformer_block_2/layer_normalization_5/moments/variance/reduction_indices?
Bmodel_2/transformer_block_2/layer_normalization_5/moments/varianceMeanOmodel_2/transformer_block_2/layer_normalization_5/moments/SquaredDifference:z:0]model_2/transformer_block_2/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2D
Bmodel_2/transformer_block_2/layer_normalization_5/moments/variance?
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add/y?
?model_2/transformer_block_2/layer_normalization_5/batchnorm/addAddV2Kmodel_2/transformer_block_2/layer_normalization_5/moments/variance:output:0Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2A
?model_2/transformer_block_2/layer_normalization_5/batchnorm/add?
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/RsqrtRsqrtCmodel_2/transformer_block_2/layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/Rsqrt?
Nmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpWmodel_2_transformer_block_2_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02P
Nmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp?
?model_2/transformer_block_2/layer_normalization_5/batchnorm/mulMulEmodel_2/transformer_block_2/layer_normalization_5/batchnorm/Rsqrt:y:0Vmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2A
?model_2/transformer_block_2/layer_normalization_5/batchnorm/mul?
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_1Mul%model_2/transformer_block_2/add_1:z:0Cmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_1?
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_2MulGmodel_2/transformer_block_2/layer_normalization_5/moments/mean:output:0Cmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_2?
Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOpSmodel_2_transformer_block_2_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02L
Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp?
?model_2/transformer_block_2/layer_normalization_5/batchnorm/subSubRmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp:value:0Emodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2A
?model_2/transformer_block_2/layer_normalization_5/batchnorm/sub?
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add_1AddV2Emodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul_1:z:0Cmodel_2/transformer_block_2/layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2C
Amodel_2/transformer_block_2/layer_normalization_5/batchnorm/add_1?
9model_2/global_average_pooling1d_2/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9model_2/global_average_pooling1d_2/Mean/reduction_indices?
'model_2/global_average_pooling1d_2/MeanMeanEmodel_2/transformer_block_2/layer_normalization_5/batchnorm/add_1:z:0Bmodel_2/global_average_pooling1d_2/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2)
'model_2/global_average_pooling1d_2/Mean?
(model_2/aux_output/MatMul/ReadVariableOpReadVariableOp1model_2_aux_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(model_2/aux_output/MatMul/ReadVariableOp?
model_2/aux_output/MatMulMatMul0model_2/global_average_pooling1d_2/Mean:output:00model_2/aux_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/aux_output/MatMul?
)model_2/aux_output/BiasAdd/ReadVariableOpReadVariableOp2model_2_aux_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_2/aux_output/BiasAdd/ReadVariableOp?
model_2/aux_output/BiasAddBiasAdd#model_2/aux_output/MatMul:product:01model_2/aux_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/aux_output/BiasAdd?
model_2/aux_output/SigmoidSigmoid#model_2/aux_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_2/aux_output/Sigmoid?
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_2/concatenate_2/concat/axis?
model_2/concatenate_2/concatConcatV2model_2/aux_output/Sigmoid:y:0	aux_input*model_2/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model_2/concatenate_2/concat?
&model_2/dense_24/MatMul/ReadVariableOpReadVariableOp/model_2_dense_24_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_2/dense_24/MatMul/ReadVariableOp?
model_2/dense_24/MatMulMatMul%model_2/concatenate_2/concat:output:0.model_2/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_24/MatMul?
'model_2/dense_24/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/dense_24/BiasAdd/ReadVariableOp?
model_2/dense_24/BiasAddBiasAdd!model_2/dense_24/MatMul:product:0/model_2/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_24/BiasAdd?
model_2/dense_24/ReluRelu!model_2/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_24/Relu?
&model_2/dense_25/MatMul/ReadVariableOpReadVariableOp/model_2_dense_25_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_2/dense_25/MatMul/ReadVariableOp?
model_2/dense_25/MatMulMatMul#model_2/dense_24/Relu:activations:0.model_2/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_25/MatMul?
'model_2/dense_25/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/dense_25/BiasAdd/ReadVariableOp?
model_2/dense_25/BiasAddBiasAdd!model_2/dense_25/MatMul:product:0/model_2/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_25/BiasAdd?
model_2/dense_25/ReluRelu!model_2/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_25/Relu?
&model_2/dense_26/MatMul/ReadVariableOpReadVariableOp/model_2_dense_26_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_2/dense_26/MatMul/ReadVariableOp?
model_2/dense_26/MatMulMatMul#model_2/dense_25/Relu:activations:0.model_2/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_26/MatMul?
'model_2/dense_26/BiasAdd/ReadVariableOpReadVariableOp0model_2_dense_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_2/dense_26/BiasAdd/ReadVariableOp?
model_2/dense_26/BiasAddBiasAdd!model_2/dense_26/MatMul:product:0/model_2/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_26/BiasAdd?
model_2/dense_26/ReluRelu!model_2/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_2/dense_26/Relu?
)model_2/main_output/MatMul/ReadVariableOpReadVariableOp2model_2_main_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02+
)model_2/main_output/MatMul/ReadVariableOp?
model_2/main_output/MatMulMatMul#model_2/dense_26/Relu:activations:01model_2/main_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/main_output/MatMul?
*model_2/main_output/BiasAdd/ReadVariableOpReadVariableOp3model_2_main_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_2/main_output/BiasAdd/ReadVariableOp?
model_2/main_output/BiasAddBiasAdd$model_2/main_output/MatMul:product:02model_2/main_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/main_output/BiasAdd?
model_2/main_output/SigmoidSigmoid$model_2/main_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_2/main_output/Sigmoidy
IdentityIdentitymodel_2/aux_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity~

Identity_1Identitymodel_2/main_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp*^model_2/aux_output/BiasAdd/ReadVariableOp)^model_2/aux_output/MatMul/ReadVariableOp(^model_2/dense_24/BiasAdd/ReadVariableOp'^model_2/dense_24/MatMul/ReadVariableOp(^model_2/dense_25/BiasAdd/ReadVariableOp'^model_2/dense_25/MatMul/ReadVariableOp(^model_2/dense_26/BiasAdd/ReadVariableOp'^model_2/dense_26/MatMul/ReadVariableOp+^model_2/main_output/BiasAdd/ReadVariableOp*^model_2/main_output/MatMul/ReadVariableOpD^model_2/token_and_position_embedding_2/embedding_4/embedding_lookupD^model_2/token_and_position_embedding_2/embedding_5/embedding_lookupK^model_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpO^model_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpK^model_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpO^model_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpX^model_2/transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpZ^model_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpX^model_2/transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpZ^model_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpX^model_2/transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpZ^model_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpX^model_2/transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpZ^model_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpI^model_2/transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOpK^model_2/transformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOpI^model_2/transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOpK^model_2/transformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)model_2/aux_output/BiasAdd/ReadVariableOp)model_2/aux_output/BiasAdd/ReadVariableOp2T
(model_2/aux_output/MatMul/ReadVariableOp(model_2/aux_output/MatMul/ReadVariableOp2R
'model_2/dense_24/BiasAdd/ReadVariableOp'model_2/dense_24/BiasAdd/ReadVariableOp2P
&model_2/dense_24/MatMul/ReadVariableOp&model_2/dense_24/MatMul/ReadVariableOp2R
'model_2/dense_25/BiasAdd/ReadVariableOp'model_2/dense_25/BiasAdd/ReadVariableOp2P
&model_2/dense_25/MatMul/ReadVariableOp&model_2/dense_25/MatMul/ReadVariableOp2R
'model_2/dense_26/BiasAdd/ReadVariableOp'model_2/dense_26/BiasAdd/ReadVariableOp2P
&model_2/dense_26/MatMul/ReadVariableOp&model_2/dense_26/MatMul/ReadVariableOp2X
*model_2/main_output/BiasAdd/ReadVariableOp*model_2/main_output/BiasAdd/ReadVariableOp2V
)model_2/main_output/MatMul/ReadVariableOp)model_2/main_output/MatMul/ReadVariableOp2?
Cmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookupCmodel_2/token_and_position_embedding_2/embedding_4/embedding_lookup2?
Cmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookupCmodel_2/token_and_position_embedding_2/embedding_5/embedding_lookup2?
Jmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOpJmodel_2/transformer_block_2/layer_normalization_4/batchnorm/ReadVariableOp2?
Nmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOpNmodel_2/transformer_block_2/layer_normalization_4/batchnorm/mul/ReadVariableOp2?
Jmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOpJmodel_2/transformer_block_2/layer_normalization_5/batchnorm/ReadVariableOp2?
Nmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOpNmodel_2/transformer_block_2/layer_normalization_5/batchnorm/mul/ReadVariableOp2?
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpWmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp2?
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpYmodel_2/transformer_block_2/multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp2?
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpWmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp2?
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpYmodel_2/transformer_block_2/multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp2?
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpWmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp2?
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpYmodel_2/transformer_block_2/multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp2?
Wmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpWmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp2?
Ymodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpYmodel_2/transformer_block_2/multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp2?
Hmodel_2/transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOpHmodel_2/transformer_block_2/sequential_2/dense_22/BiasAdd/ReadVariableOp2?
Jmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOpJmodel_2/transformer_block_2/sequential_2/dense_22/Tensordot/ReadVariableOp2?
Hmodel_2/transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOpHmodel_2/transformer_block_2/sequential_2/dense_23/BiasAdd/ReadVariableOp2?
Jmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOpJmodel_2/transformer_block_2/sequential_2/dense_23/Tensordot/ReadVariableOp:P L
'
_output_shapes
:?????????(
!
_user_specified_name	input_3:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input
?
?
E__inference_dense_25_layer_call_and_return_conditional_losses_1351114

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
?
?
*__inference_dense_26_layer_call_fn_1353564

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
GPU 2J 8? *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_13511312
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
?
[
/__inference_concatenate_2_layer_call_fn_1353508
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_13510842
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
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
inputs/1
?
?
*__inference_dense_25_layer_call_fn_1353544

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
GPU 2J 8? *N
fIRG
E__inference_dense_25_layer_call_and_return_conditional_losses_13511142
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
?
?
E__inference_dense_25_layer_call_and_return_conditional_losses_1353555

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
?
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1350712
dense_22_input"
dense_22_1350701:  
dense_22_1350703: "
dense_23_1350706:  
dense_23_1350708: 
identity?? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
 dense_22/StatefulPartitionedCallStatefulPartitionedCalldense_22_inputdense_22_1350701dense_22_1350703*
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
GPU 2J 8? *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_13505572"
 dense_22/StatefulPartitionedCall?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_1350706dense_23_1350708*
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
GPU 2J 8? *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_13505932"
 dense_23/StatefulPartitionedCall?
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:[ W
+
_output_shapes
:?????????( 
(
_user_specified_namedense_22_input
?
?
@__inference_token_and_position_embedding_2_layer_call_fn_1352860
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
GPU 2J 8? *d
f_R]
[__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_13507692
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
?
?
)__inference_model_2_layer_call_fn_1352227
inputs_0
inputs_1
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

unknown_19:@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@

unknown_26:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_13517602
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
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
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
inputs/1
?
?
.__inference_sequential_2_layer_call_fn_1350684
dense_22_input
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_22_inputunknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8? *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_13506602
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
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:?????????( 
(
_user_specified_namedense_22_input
?K
?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1353678

inputs<
*dense_22_tensordot_readvariableop_resource:  6
(dense_22_biasadd_readvariableop_resource: <
*dense_23_tensordot_readvariableop_resource:  6
(dense_23_biasadd_readvariableop_resource: 
identity??dense_22/BiasAdd/ReadVariableOp?!dense_22/Tensordot/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?!dense_23/Tensordot/ReadVariableOp?
!dense_22/Tensordot/ReadVariableOpReadVariableOp*dense_22_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_22/Tensordot/ReadVariableOp|
dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_22/Tensordot/axes?
dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_22/Tensordot/freej
dense_22/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_22/Tensordot/Shape?
 dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_22/Tensordot/GatherV2/axis?
dense_22/Tensordot/GatherV2GatherV2!dense_22/Tensordot/Shape:output:0 dense_22/Tensordot/free:output:0)dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_22/Tensordot/GatherV2?
"dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_22/Tensordot/GatherV2_1/axis?
dense_22/Tensordot/GatherV2_1GatherV2!dense_22/Tensordot/Shape:output:0 dense_22/Tensordot/axes:output:0+dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_22/Tensordot/GatherV2_1~
dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_22/Tensordot/Const?
dense_22/Tensordot/ProdProd$dense_22/Tensordot/GatherV2:output:0!dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_22/Tensordot/Prod?
dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_22/Tensordot/Const_1?
dense_22/Tensordot/Prod_1Prod&dense_22/Tensordot/GatherV2_1:output:0#dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_22/Tensordot/Prod_1?
dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_22/Tensordot/concat/axis?
dense_22/Tensordot/concatConcatV2 dense_22/Tensordot/free:output:0 dense_22/Tensordot/axes:output:0'dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/concat?
dense_22/Tensordot/stackPack dense_22/Tensordot/Prod:output:0"dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/stack?
dense_22/Tensordot/transpose	Transposeinputs"dense_22/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_22/Tensordot/transpose?
dense_22/Tensordot/ReshapeReshape dense_22/Tensordot/transpose:y:0!dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_22/Tensordot/Reshape?
dense_22/Tensordot/MatMulMatMul#dense_22/Tensordot/Reshape:output:0)dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_22/Tensordot/MatMul?
dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_22/Tensordot/Const_2?
 dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_22/Tensordot/concat_1/axis?
dense_22/Tensordot/concat_1ConcatV2$dense_22/Tensordot/GatherV2:output:0#dense_22/Tensordot/Const_2:output:0)dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_22/Tensordot/concat_1?
dense_22/TensordotReshape#dense_22/Tensordot/MatMul:product:0$dense_22/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_22/Tensordot?
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_22/BiasAdd/ReadVariableOp?
dense_22/BiasAddBiasAdddense_22/Tensordot:output:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_22/BiasAddw
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dense_22/Relu?
!dense_23/Tensordot/ReadVariableOpReadVariableOp*dense_23_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_23/Tensordot/ReadVariableOp|
dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_23/Tensordot/axes?
dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_23/Tensordot/free
dense_23/Tensordot/ShapeShapedense_22/Relu:activations:0*
T0*
_output_shapes
:2
dense_23/Tensordot/Shape?
 dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/GatherV2/axis?
dense_23/Tensordot/GatherV2GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/free:output:0)dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2?
"dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_23/Tensordot/GatherV2_1/axis?
dense_23/Tensordot/GatherV2_1GatherV2!dense_23/Tensordot/Shape:output:0 dense_23/Tensordot/axes:output:0+dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_23/Tensordot/GatherV2_1~
dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const?
dense_23/Tensordot/ProdProd$dense_23/Tensordot/GatherV2:output:0!dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod?
dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const_1?
dense_23/Tensordot/Prod_1Prod&dense_23/Tensordot/GatherV2_1:output:0#dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_23/Tensordot/Prod_1?
dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_23/Tensordot/concat/axis?
dense_23/Tensordot/concatConcatV2 dense_23/Tensordot/free:output:0 dense_23/Tensordot/axes:output:0'dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat?
dense_23/Tensordot/stackPack dense_23/Tensordot/Prod:output:0"dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/stack?
dense_23/Tensordot/transpose	Transposedense_22/Relu:activations:0"dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_23/Tensordot/transpose?
dense_23/Tensordot/ReshapeReshape dense_23/Tensordot/transpose:y:0!dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_23/Tensordot/Reshape?
dense_23/Tensordot/MatMulMatMul#dense_23/Tensordot/Reshape:output:0)dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_23/Tensordot/MatMul?
dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_23/Tensordot/Const_2?
 dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_23/Tensordot/concat_1/axis?
dense_23/Tensordot/concat_1ConcatV2$dense_23/Tensordot/GatherV2:output:0#dense_23/Tensordot/Const_2:output:0)dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_23/Tensordot/concat_1?
dense_23/TensordotReshape#dense_23/Tensordot/MatMul:product:0$dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_23/Tensordot?
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_23/BiasAdd/ReadVariableOp?
dense_23/BiasAddBiasAdddense_23/Tensordot:output:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_23/BiasAddx
IdentityIdentitydense_23/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp ^dense_22/BiasAdd/ReadVariableOp"^dense_22/Tensordot/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp"^dense_23/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2F
!dense_22/Tensordot/ReadVariableOp!dense_22/Tensordot/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2F
!dense_23/Tensordot/ReadVariableOp!dense_23/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
[__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_1350769
x6
$embedding_5_embedding_lookup_1350756:( 6
$embedding_4_embedding_lookup_1350762: 
identity??embedding_4/embedding_lookup?embedding_5/embedding_lookup?
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
embedding_5/embedding_lookupResourceGather$embedding_5_embedding_lookup_1350756range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_5/embedding_lookup/1350756*'
_output_shapes
:????????? *
dtype02
embedding_5/embedding_lookup?
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_5/embedding_lookup/1350756*'
_output_shapes
:????????? 2'
%embedding_5/embedding_lookup/Identity?
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_5/embedding_lookup/Identity_1p
embedding_4/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:?????????(2
embedding_4/Cast?
embedding_4/embedding_lookupResourceGather$embedding_4_embedding_lookup_1350762embedding_4/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_4/embedding_lookup/1350762*+
_output_shapes
:?????????( *
dtype02
embedding_4/embedding_lookup?
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_4/embedding_lookup/1350762*+
_output_shapes
:?????????( 2'
%embedding_4/embedding_lookup/Identity?
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2)
'embedding_4/embedding_lookup/Identity_1?
addAddV20embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2
addf
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp^embedding_4/embedding_lookup^embedding_5/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup:J F
'
_output_shapes
:?????????(

_user_specified_namex
??
?
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_1353460

inputsX
Fmulti_head_self_attention_2_dense_18_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_18_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_19_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_19_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_20_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_20_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_2_dense_21_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_2_dense_21_biasadd_readvariableop_resource: I
;layer_normalization_4_batchnorm_mul_readvariableop_resource: E
7layer_normalization_4_batchnorm_readvariableop_resource: I
7sequential_2_dense_22_tensordot_readvariableop_resource:  C
5sequential_2_dense_22_biasadd_readvariableop_resource: I
7sequential_2_dense_23_tensordot_readvariableop_resource:  C
5sequential_2_dense_23_biasadd_readvariableop_resource: I
;layer_normalization_5_batchnorm_mul_readvariableop_resource: E
7layer_normalization_5_batchnorm_readvariableop_resource: 
identity??.layer_normalization_4/batchnorm/ReadVariableOp?2layer_normalization_4/batchnorm/mul/ReadVariableOp?.layer_normalization_5/batchnorm/ReadVariableOp?2layer_normalization_5/batchnorm/mul/ReadVariableOp?;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?,sequential_2/dense_22/BiasAdd/ReadVariableOp?.sequential_2/dense_22/Tensordot/ReadVariableOp?,sequential_2/dense_23/BiasAdd/ReadVariableOp?.sequential_2/dense_23/Tensordot/ReadVariableOp|
!multi_head_self_attention_2/ShapeShapeinputs*
T0*
_output_shapes
:2#
!multi_head_self_attention_2/Shape?
/multi_head_self_attention_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention_2/strided_slice/stack?
1multi_head_self_attention_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_2/strided_slice/stack_1?
1multi_head_self_attention_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_2/strided_slice/stack_2?
)multi_head_self_attention_2/strided_sliceStridedSlice*multi_head_self_attention_2/Shape:output:08multi_head_self_attention_2/strided_slice/stack:output:0:multi_head_self_attention_2/strided_slice/stack_1:output:0:multi_head_self_attention_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention_2/strided_slice?
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_18_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_18/Tensordot/axes?
3multi_head_self_attention_2/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_18/Tensordot/free?
4multi_head_self_attention_2/dense_18/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_18/Tensordot/Shape?
<multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_18/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_18/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_18/Tensordot/free:output:0Emulti_head_self_attention_2/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_18/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_18/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_18/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_18/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_18/Tensordot/Const?
3multi_head_self_attention_2/dense_18/Tensordot/ProdProd@multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_18/Tensordot/Prod?
6multi_head_self_attention_2/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_18/Tensordot/Const_1?
5multi_head_self_attention_2/dense_18/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_18/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_18/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_18/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_18/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_18/Tensordot/free:output:0<multi_head_self_attention_2/dense_18/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_18/Tensordot/concat?
4multi_head_self_attention_2/dense_18/Tensordot/stackPack<multi_head_self_attention_2/dense_18/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_18/Tensordot/stack?
8multi_head_self_attention_2/dense_18/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_18/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_18/Tensordot/transpose?
6multi_head_self_attention_2/dense_18/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_18/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_18/Tensordot/Reshape?
5multi_head_self_attention_2/dense_18/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_18/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_18/Tensordot/MatMul?
6multi_head_self_attention_2/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_18/Tensordot/Const_2?
<multi_head_self_attention_2/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_18/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_18/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_18/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_18/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_18/Tensordot/concat_1?
.multi_head_self_attention_2/dense_18/TensordotReshape?multi_head_self_attention_2/dense_18/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_18/Tensordot?
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_18/BiasAddBiasAdd7multi_head_self_attention_2/dense_18/Tensordot:output:0Cmulti_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_18/BiasAdd?
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_19_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_19/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_19/Tensordot/axes?
3multi_head_self_attention_2/dense_19/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_19/Tensordot/free?
4multi_head_self_attention_2/dense_19/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_19/Tensordot/Shape?
<multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_19/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_19/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_19/Tensordot/free:output:0Emulti_head_self_attention_2/dense_19/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_19/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_19/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_19/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_19/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_19/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_19/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_19/Tensordot/Const?
3multi_head_self_attention_2/dense_19/Tensordot/ProdProd@multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_19/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_19/Tensordot/Prod?
6multi_head_self_attention_2/dense_19/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_19/Tensordot/Const_1?
5multi_head_self_attention_2/dense_19/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_19/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_19/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_19/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_19/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_19/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_19/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_19/Tensordot/free:output:0<multi_head_self_attention_2/dense_19/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_19/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_19/Tensordot/concat?
4multi_head_self_attention_2/dense_19/Tensordot/stackPack<multi_head_self_attention_2/dense_19/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_19/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_19/Tensordot/stack?
8multi_head_self_attention_2/dense_19/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_19/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_19/Tensordot/transpose?
6multi_head_self_attention_2/dense_19/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_19/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_19/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_19/Tensordot/Reshape?
5multi_head_self_attention_2/dense_19/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_19/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_19/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_19/Tensordot/MatMul?
6multi_head_self_attention_2/dense_19/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_19/Tensordot/Const_2?
<multi_head_self_attention_2/dense_19/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_19/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_19/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_19/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_19/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_19/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_19/Tensordot/concat_1?
.multi_head_self_attention_2/dense_19/TensordotReshape?multi_head_self_attention_2/dense_19/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_19/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_19/Tensordot?
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_19/BiasAddBiasAdd7multi_head_self_attention_2/dense_19/Tensordot:output:0Cmulti_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_19/BiasAdd?
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_20_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_20/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_20/Tensordot/axes?
3multi_head_self_attention_2/dense_20/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_20/Tensordot/free?
4multi_head_self_attention_2/dense_20/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_20/Tensordot/Shape?
<multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_20/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_20/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_20/Tensordot/free:output:0Emulti_head_self_attention_2/dense_20/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_20/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_20/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_20/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_20/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_20/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_20/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_20/Tensordot/Const?
3multi_head_self_attention_2/dense_20/Tensordot/ProdProd@multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_20/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_20/Tensordot/Prod?
6multi_head_self_attention_2/dense_20/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_20/Tensordot/Const_1?
5multi_head_self_attention_2/dense_20/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_20/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_20/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_20/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_20/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_20/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_20/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_20/Tensordot/free:output:0<multi_head_self_attention_2/dense_20/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_20/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_20/Tensordot/concat?
4multi_head_self_attention_2/dense_20/Tensordot/stackPack<multi_head_self_attention_2/dense_20/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_20/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_20/Tensordot/stack?
8multi_head_self_attention_2/dense_20/Tensordot/transpose	Transposeinputs>multi_head_self_attention_2/dense_20/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_2/dense_20/Tensordot/transpose?
6multi_head_self_attention_2/dense_20/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_20/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_20/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_20/Tensordot/Reshape?
5multi_head_self_attention_2/dense_20/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_20/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_20/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_20/Tensordot/MatMul?
6multi_head_self_attention_2/dense_20/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_20/Tensordot/Const_2?
<multi_head_self_attention_2/dense_20/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_20/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_20/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_20/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_20/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_20/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_20/Tensordot/concat_1?
.multi_head_self_attention_2/dense_20/TensordotReshape?multi_head_self_attention_2/dense_20/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_20/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_2/dense_20/Tensordot?
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_20/BiasAddBiasAdd7multi_head_self_attention_2/dense_20/Tensordot:output:0Cmulti_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_2/dense_20/BiasAdd?
+multi_head_self_attention_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+multi_head_self_attention_2/Reshape/shape/1?
+multi_head_self_attention_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_2/Reshape/shape/2?
+multi_head_self_attention_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_2/Reshape/shape/3?
)multi_head_self_attention_2/Reshape/shapePack2multi_head_self_attention_2/strided_slice:output:04multi_head_self_attention_2/Reshape/shape/1:output:04multi_head_self_attention_2/Reshape/shape/2:output:04multi_head_self_attention_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention_2/Reshape/shape?
#multi_head_self_attention_2/ReshapeReshape5multi_head_self_attention_2/dense_18/BiasAdd:output:02multi_head_self_attention_2/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2%
#multi_head_self_attention_2/Reshape?
*multi_head_self_attention_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention_2/transpose/perm?
%multi_head_self_attention_2/transpose	Transpose,multi_head_self_attention_2/Reshape:output:03multi_head_self_attention_2/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/transpose?
-multi_head_self_attention_2/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_1/shape/1?
-multi_head_self_attention_2/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_1/shape/2?
-multi_head_self_attention_2/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_1/shape/3?
+multi_head_self_attention_2/Reshape_1/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_1/shape/1:output:06multi_head_self_attention_2/Reshape_1/shape/2:output:06multi_head_self_attention_2/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_1/shape?
%multi_head_self_attention_2/Reshape_1Reshape5multi_head_self_attention_2/dense_19/BiasAdd:output:04multi_head_self_attention_2/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/Reshape_1?
,multi_head_self_attention_2/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_1/perm?
'multi_head_self_attention_2/transpose_1	Transpose.multi_head_self_attention_2/Reshape_1:output:05multi_head_self_attention_2/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_1?
-multi_head_self_attention_2/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_2/shape/1?
-multi_head_self_attention_2/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_2/shape/2?
-multi_head_self_attention_2/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_2/Reshape_2/shape/3?
+multi_head_self_attention_2/Reshape_2/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_2/shape/1:output:06multi_head_self_attention_2/Reshape_2/shape/2:output:06multi_head_self_attention_2/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_2/shape?
%multi_head_self_attention_2/Reshape_2Reshape5multi_head_self_attention_2/dense_20/BiasAdd:output:04multi_head_self_attention_2/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_2/Reshape_2?
,multi_head_self_attention_2/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_2/perm?
'multi_head_self_attention_2/transpose_2	Transpose.multi_head_self_attention_2/Reshape_2:output:05multi_head_self_attention_2/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_2?
"multi_head_self_attention_2/MatMulBatchMatMulV2)multi_head_self_attention_2/transpose:y:0+multi_head_self_attention_2/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2$
"multi_head_self_attention_2/MatMul?
#multi_head_self_attention_2/Shape_1Shape+multi_head_self_attention_2/transpose_1:y:0*
T0*
_output_shapes
:2%
#multi_head_self_attention_2/Shape_1?
1multi_head_self_attention_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1multi_head_self_attention_2/strided_slice_1/stack?
3multi_head_self_attention_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention_2/strided_slice_1/stack_1?
3multi_head_self_attention_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/strided_slice_1/stack_2?
+multi_head_self_attention_2/strided_slice_1StridedSlice,multi_head_self_attention_2/Shape_1:output:0:multi_head_self_attention_2/strided_slice_1/stack:output:0<multi_head_self_attention_2/strided_slice_1/stack_1:output:0<multi_head_self_attention_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+multi_head_self_attention_2/strided_slice_1?
 multi_head_self_attention_2/CastCast4multi_head_self_attention_2/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 multi_head_self_attention_2/Cast?
 multi_head_self_attention_2/SqrtSqrt$multi_head_self_attention_2/Cast:y:0*
T0*
_output_shapes
: 2"
 multi_head_self_attention_2/Sqrt?
#multi_head_self_attention_2/truedivRealDiv+multi_head_self_attention_2/MatMul:output:0$multi_head_self_attention_2/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_2/truediv?
#multi_head_self_attention_2/SoftmaxSoftmax'multi_head_self_attention_2/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_2/Softmax?
$multi_head_self_attention_2/MatMul_1BatchMatMulV2-multi_head_self_attention_2/Softmax:softmax:0+multi_head_self_attention_2/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_2/MatMul_1?
,multi_head_self_attention_2/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_2/transpose_3/perm?
'multi_head_self_attention_2/transpose_3	Transpose-multi_head_self_attention_2/MatMul_1:output:05multi_head_self_attention_2/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_2/transpose_3?
-multi_head_self_attention_2/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_2/Reshape_3/shape/1?
-multi_head_self_attention_2/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2/
-multi_head_self_attention_2/Reshape_3/shape/2?
+multi_head_self_attention_2/Reshape_3/shapePack2multi_head_self_attention_2/strided_slice:output:06multi_head_self_attention_2/Reshape_3/shape/1:output:06multi_head_self_attention_2/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_2/Reshape_3/shape?
%multi_head_self_attention_2/Reshape_3Reshape+multi_head_self_attention_2/transpose_3:y:04multi_head_self_attention_2/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2'
%multi_head_self_attention_2/Reshape_3?
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_2_dense_21_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp?
3multi_head_self_attention_2/dense_21/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_2/dense_21/Tensordot/axes?
3multi_head_self_attention_2/dense_21/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_2/dense_21/Tensordot/free?
4multi_head_self_attention_2/dense_21/Tensordot/ShapeShape.multi_head_self_attention_2/Reshape_3:output:0*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_21/Tensordot/Shape?
<multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_21/Tensordot/GatherV2/axis?
7multi_head_self_attention_2/dense_21/Tensordot/GatherV2GatherV2=multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_21/Tensordot/free:output:0Emulti_head_self_attention_2/dense_21/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_2/dense_21/Tensordot/GatherV2?
>multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_2/dense_21/Tensordot/Shape:output:0<multi_head_self_attention_2/dense_21/Tensordot/axes:output:0Gmulti_head_self_attention_2/dense_21/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_2/dense_21/Tensordot/GatherV2_1?
4multi_head_self_attention_2/dense_21/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_2/dense_21/Tensordot/Const?
3multi_head_self_attention_2/dense_21/Tensordot/ProdProd@multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0=multi_head_self_attention_2/dense_21/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_2/dense_21/Tensordot/Prod?
6multi_head_self_attention_2/dense_21/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_21/Tensordot/Const_1?
5multi_head_self_attention_2/dense_21/Tensordot/Prod_1ProdBmulti_head_self_attention_2/dense_21/Tensordot/GatherV2_1:output:0?multi_head_self_attention_2/dense_21/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_2/dense_21/Tensordot/Prod_1?
:multi_head_self_attention_2/dense_21/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_2/dense_21/Tensordot/concat/axis?
5multi_head_self_attention_2/dense_21/Tensordot/concatConcatV2<multi_head_self_attention_2/dense_21/Tensordot/free:output:0<multi_head_self_attention_2/dense_21/Tensordot/axes:output:0Cmulti_head_self_attention_2/dense_21/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_2/dense_21/Tensordot/concat?
4multi_head_self_attention_2/dense_21/Tensordot/stackPack<multi_head_self_attention_2/dense_21/Tensordot/Prod:output:0>multi_head_self_attention_2/dense_21/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_2/dense_21/Tensordot/stack?
8multi_head_self_attention_2/dense_21/Tensordot/transpose	Transpose.multi_head_self_attention_2/Reshape_3:output:0>multi_head_self_attention_2/dense_21/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2:
8multi_head_self_attention_2/dense_21/Tensordot/transpose?
6multi_head_self_attention_2/dense_21/Tensordot/ReshapeReshape<multi_head_self_attention_2/dense_21/Tensordot/transpose:y:0=multi_head_self_attention_2/dense_21/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_2/dense_21/Tensordot/Reshape?
5multi_head_self_attention_2/dense_21/Tensordot/MatMulMatMul?multi_head_self_attention_2/dense_21/Tensordot/Reshape:output:0Emulti_head_self_attention_2/dense_21/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_2/dense_21/Tensordot/MatMul?
6multi_head_self_attention_2/dense_21/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_2/dense_21/Tensordot/Const_2?
<multi_head_self_attention_2/dense_21/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_2/dense_21/Tensordot/concat_1/axis?
7multi_head_self_attention_2/dense_21/Tensordot/concat_1ConcatV2@multi_head_self_attention_2/dense_21/Tensordot/GatherV2:output:0?multi_head_self_attention_2/dense_21/Tensordot/Const_2:output:0Emulti_head_self_attention_2/dense_21/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_2/dense_21/Tensordot/concat_1?
.multi_head_self_attention_2/dense_21/TensordotReshape?multi_head_self_attention_2/dense_21/Tensordot/MatMul:product:0@multi_head_self_attention_2/dense_21/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_2/dense_21/Tensordot?
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_2_dense_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp?
,multi_head_self_attention_2/dense_21/BiasAddBiasAdd7multi_head_self_attention_2/dense_21/Tensordot:output:0Cmulti_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2.
,multi_head_self_attention_2/dense_21/BiasAddw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_4/dropout/Const?
dropout_4/dropout/MulMul5multi_head_self_attention_2/dense_21/BiasAdd:output:0 dropout_4/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_4/dropout/Mul?
dropout_4/dropout/ShapeShape5multi_head_self_attention_2/dense_21/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype020
.dropout_4/dropout/random_uniform/RandomUniform?
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_4/dropout/GreaterEqual/y?
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2 
dropout_4/dropout/GreaterEqual?
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout_4/dropout/Cast?
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_4/dropout/Mul_1n
addAddV2inputsdropout_4/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add?
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indices?
"layer_normalization_4/moments/meanMeanadd:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2$
"layer_normalization_4/moments/mean?
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2,
*layer_normalization_4/moments/StopGradient?
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 21
/layer_normalization_4/moments/SquaredDifference?
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indices?
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2(
&layer_normalization_4/moments/variance?
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_4/batchnorm/add/y?
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2%
#layer_normalization_4/batchnorm/add?
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2'
%layer_normalization_4/batchnorm/Rsqrt?
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOp?
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_4/batchnorm/mul?
%layer_normalization_4/batchnorm/mul_1Muladd:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/mul_1?
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/mul_2?
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_4/batchnorm/ReadVariableOp?
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_4/batchnorm/sub?
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_4/batchnorm/add_1?
.sequential_2/dense_22/Tensordot/ReadVariableOpReadVariableOp7sequential_2_dense_22_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_2/dense_22/Tensordot/ReadVariableOp?
$sequential_2/dense_22/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_2/dense_22/Tensordot/axes?
$sequential_2/dense_22/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_2/dense_22/Tensordot/free?
%sequential_2/dense_22/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_2/dense_22/Tensordot/Shape?
-sequential_2/dense_22/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_22/Tensordot/GatherV2/axis?
(sequential_2/dense_22/Tensordot/GatherV2GatherV2.sequential_2/dense_22/Tensordot/Shape:output:0-sequential_2/dense_22/Tensordot/free:output:06sequential_2/dense_22/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_2/dense_22/Tensordot/GatherV2?
/sequential_2/dense_22/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_2/dense_22/Tensordot/GatherV2_1/axis?
*sequential_2/dense_22/Tensordot/GatherV2_1GatherV2.sequential_2/dense_22/Tensordot/Shape:output:0-sequential_2/dense_22/Tensordot/axes:output:08sequential_2/dense_22/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_2/dense_22/Tensordot/GatherV2_1?
%sequential_2/dense_22/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_2/dense_22/Tensordot/Const?
$sequential_2/dense_22/Tensordot/ProdProd1sequential_2/dense_22/Tensordot/GatherV2:output:0.sequential_2/dense_22/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_2/dense_22/Tensordot/Prod?
'sequential_2/dense_22/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_22/Tensordot/Const_1?
&sequential_2/dense_22/Tensordot/Prod_1Prod3sequential_2/dense_22/Tensordot/GatherV2_1:output:00sequential_2/dense_22/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_2/dense_22/Tensordot/Prod_1?
+sequential_2/dense_22/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_2/dense_22/Tensordot/concat/axis?
&sequential_2/dense_22/Tensordot/concatConcatV2-sequential_2/dense_22/Tensordot/free:output:0-sequential_2/dense_22/Tensordot/axes:output:04sequential_2/dense_22/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_2/dense_22/Tensordot/concat?
%sequential_2/dense_22/Tensordot/stackPack-sequential_2/dense_22/Tensordot/Prod:output:0/sequential_2/dense_22/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_22/Tensordot/stack?
)sequential_2/dense_22/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0/sequential_2/dense_22/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_2/dense_22/Tensordot/transpose?
'sequential_2/dense_22/Tensordot/ReshapeReshape-sequential_2/dense_22/Tensordot/transpose:y:0.sequential_2/dense_22/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_2/dense_22/Tensordot/Reshape?
&sequential_2/dense_22/Tensordot/MatMulMatMul0sequential_2/dense_22/Tensordot/Reshape:output:06sequential_2/dense_22/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_2/dense_22/Tensordot/MatMul?
'sequential_2/dense_22/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_22/Tensordot/Const_2?
-sequential_2/dense_22/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_22/Tensordot/concat_1/axis?
(sequential_2/dense_22/Tensordot/concat_1ConcatV21sequential_2/dense_22/Tensordot/GatherV2:output:00sequential_2/dense_22/Tensordot/Const_2:output:06sequential_2/dense_22/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_2/dense_22/Tensordot/concat_1?
sequential_2/dense_22/TensordotReshape0sequential_2/dense_22/Tensordot/MatMul:product:01sequential_2/dense_22/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_2/dense_22/Tensordot?
,sequential_2/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/dense_22/BiasAdd/ReadVariableOp?
sequential_2/dense_22/BiasAddBiasAdd(sequential_2/dense_22/Tensordot:output:04sequential_2/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_22/BiasAdd?
sequential_2/dense_22/ReluRelu&sequential_2/dense_22/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_22/Relu?
.sequential_2/dense_23/Tensordot/ReadVariableOpReadVariableOp7sequential_2_dense_23_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_2/dense_23/Tensordot/ReadVariableOp?
$sequential_2/dense_23/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_2/dense_23/Tensordot/axes?
$sequential_2/dense_23/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_2/dense_23/Tensordot/free?
%sequential_2/dense_23/Tensordot/ShapeShape(sequential_2/dense_22/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_2/dense_23/Tensordot/Shape?
-sequential_2/dense_23/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_23/Tensordot/GatherV2/axis?
(sequential_2/dense_23/Tensordot/GatherV2GatherV2.sequential_2/dense_23/Tensordot/Shape:output:0-sequential_2/dense_23/Tensordot/free:output:06sequential_2/dense_23/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_2/dense_23/Tensordot/GatherV2?
/sequential_2/dense_23/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_2/dense_23/Tensordot/GatherV2_1/axis?
*sequential_2/dense_23/Tensordot/GatherV2_1GatherV2.sequential_2/dense_23/Tensordot/Shape:output:0-sequential_2/dense_23/Tensordot/axes:output:08sequential_2/dense_23/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_2/dense_23/Tensordot/GatherV2_1?
%sequential_2/dense_23/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_2/dense_23/Tensordot/Const?
$sequential_2/dense_23/Tensordot/ProdProd1sequential_2/dense_23/Tensordot/GatherV2:output:0.sequential_2/dense_23/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_2/dense_23/Tensordot/Prod?
'sequential_2/dense_23/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_23/Tensordot/Const_1?
&sequential_2/dense_23/Tensordot/Prod_1Prod3sequential_2/dense_23/Tensordot/GatherV2_1:output:00sequential_2/dense_23/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_2/dense_23/Tensordot/Prod_1?
+sequential_2/dense_23/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_2/dense_23/Tensordot/concat/axis?
&sequential_2/dense_23/Tensordot/concatConcatV2-sequential_2/dense_23/Tensordot/free:output:0-sequential_2/dense_23/Tensordot/axes:output:04sequential_2/dense_23/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_2/dense_23/Tensordot/concat?
%sequential_2/dense_23/Tensordot/stackPack-sequential_2/dense_23/Tensordot/Prod:output:0/sequential_2/dense_23/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_2/dense_23/Tensordot/stack?
)sequential_2/dense_23/Tensordot/transpose	Transpose(sequential_2/dense_22/Relu:activations:0/sequential_2/dense_23/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_2/dense_23/Tensordot/transpose?
'sequential_2/dense_23/Tensordot/ReshapeReshape-sequential_2/dense_23/Tensordot/transpose:y:0.sequential_2/dense_23/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_2/dense_23/Tensordot/Reshape?
&sequential_2/dense_23/Tensordot/MatMulMatMul0sequential_2/dense_23/Tensordot/Reshape:output:06sequential_2/dense_23/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_2/dense_23/Tensordot/MatMul?
'sequential_2/dense_23/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_2/dense_23/Tensordot/Const_2?
-sequential_2/dense_23/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_2/dense_23/Tensordot/concat_1/axis?
(sequential_2/dense_23/Tensordot/concat_1ConcatV21sequential_2/dense_23/Tensordot/GatherV2:output:00sequential_2/dense_23/Tensordot/Const_2:output:06sequential_2/dense_23/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_2/dense_23/Tensordot/concat_1?
sequential_2/dense_23/TensordotReshape0sequential_2/dense_23/Tensordot/MatMul:product:01sequential_2/dense_23/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_2/dense_23/Tensordot?
,sequential_2/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_2/dense_23/BiasAdd/ReadVariableOp?
sequential_2/dense_23/BiasAddBiasAdd(sequential_2/dense_23/Tensordot:output:04sequential_2/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_2/dense_23/BiasAddw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_5/dropout/Const?
dropout_5/dropout/MulMul&sequential_2/dense_23/BiasAdd:output:0 dropout_5/dropout/Const:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_5/dropout/Mul?
dropout_5/dropout/ShapeShape&sequential_2/dense_23/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape?
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????( *
dtype020
.dropout_5/dropout/random_uniform/RandomUniform?
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_5/dropout/GreaterEqual/y?
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????( 2 
dropout_5/dropout/GreaterEqual?
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????( 2
dropout_5/dropout/Cast?
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????( 2
dropout_5/dropout/Mul_1?
add_1AddV2)layer_normalization_4/batchnorm/add_1:z:0dropout_5/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add_1?
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_5/moments/mean/reduction_indices?
"layer_normalization_5/moments/meanMean	add_1:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2$
"layer_normalization_5/moments/mean?
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2,
*layer_normalization_5/moments/StopGradient?
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference	add_1:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 21
/layer_normalization_5/moments/SquaredDifference?
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_5/moments/variance/reduction_indices?
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2(
&layer_normalization_5/moments/variance?
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52'
%layer_normalization_5/batchnorm/add/y?
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2%
#layer_normalization_5/batchnorm/add?
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2'
%layer_normalization_5/batchnorm/Rsqrt?
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2layer_normalization_5/batchnorm/mul/ReadVariableOp?
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_5/batchnorm/mul?
%layer_normalization_5/batchnorm/mul_1Mul	add_1:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/mul_1?
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/mul_2?
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.layer_normalization_5/batchnorm/ReadVariableOp?
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2%
#layer_normalization_5/batchnorm/sub?
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2'
%layer_normalization_5/batchnorm/add_1?
IdentityIdentity)layer_normalization_5/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp<^multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp<^multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp>^multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp-^sequential_2/dense_22/BiasAdd/ReadVariableOp/^sequential_2/dense_22/Tensordot/ReadVariableOp-^sequential_2/dense_23/BiasAdd/ReadVariableOp/^sequential_2/dense_23/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp2z
;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_18/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_18/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_19/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_19/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_20/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_20/Tensordot/ReadVariableOp2z
;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp;multi_head_self_attention_2/dense_21/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp=multi_head_self_attention_2/dense_21/Tensordot/ReadVariableOp2\
,sequential_2/dense_22/BiasAdd/ReadVariableOp,sequential_2/dense_22/BiasAdd/ReadVariableOp2`
.sequential_2/dense_22/Tensordot/ReadVariableOp.sequential_2/dense_22/Tensordot/ReadVariableOp2\
,sequential_2/dense_23/BiasAdd/ReadVariableOp,sequential_2/dense_23/BiasAdd/ReadVariableOp2`
.sequential_2/dense_23/Tensordot/ReadVariableOp.sequential_2/dense_23/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
??
?5
 __inference__traced_save_1354136
file_prefix0
,savev2_aux_output_kernel_read_readvariableop.
*savev2_aux_output_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop1
-savev2_main_output_kernel_read_readvariableop/
+savev2_main_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopT
Psavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopT
Psavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableop^
Zsavev2_transformer_block_2_multi_head_self_attention_2_dense_18_kernel_read_readvariableop\
Xsavev2_transformer_block_2_multi_head_self_attention_2_dense_18_bias_read_readvariableop^
Zsavev2_transformer_block_2_multi_head_self_attention_2_dense_19_kernel_read_readvariableop\
Xsavev2_transformer_block_2_multi_head_self_attention_2_dense_19_bias_read_readvariableop^
Zsavev2_transformer_block_2_multi_head_self_attention_2_dense_20_kernel_read_readvariableop\
Xsavev2_transformer_block_2_multi_head_self_attention_2_dense_20_bias_read_readvariableop^
Zsavev2_transformer_block_2_multi_head_self_attention_2_dense_21_kernel_read_readvariableop\
Xsavev2_transformer_block_2_multi_head_self_attention_2_dense_21_bias_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableopN
Jsavev2_transformer_block_2_layer_normalization_4_gamma_read_readvariableopM
Isavev2_transformer_block_2_layer_normalization_4_beta_read_readvariableopN
Jsavev2_transformer_block_2_layer_normalization_5_gamma_read_readvariableopM
Isavev2_transformer_block_2_layer_normalization_5_beta_read_readvariableop$
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
1savev2_adam_aux_output_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop5
1savev2_adam_dense_25_kernel_m_read_readvariableop3
/savev2_adam_dense_25_bias_m_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableop8
4savev2_adam_main_output_kernel_m_read_readvariableop6
2savev2_adam_main_output_bias_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_m_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_m_read_readvariableope
asavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_18_kernel_m_read_readvariableopc
_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_18_bias_m_read_readvariableope
asavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_19_kernel_m_read_readvariableopc
_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_19_bias_m_read_readvariableope
asavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_20_kernel_m_read_readvariableopc
_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_20_bias_m_read_readvariableope
asavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_21_kernel_m_read_readvariableopc
_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_21_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_4_gamma_m_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_4_beta_m_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_5_gamma_m_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_5_beta_m_read_readvariableop7
3savev2_adam_aux_output_kernel_v_read_readvariableop5
1savev2_adam_aux_output_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop5
1savev2_adam_dense_25_kernel_v_read_readvariableop3
/savev2_adam_dense_25_bias_v_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableop8
4savev2_adam_main_output_kernel_v_read_readvariableop6
2savev2_adam_main_output_bias_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_v_read_readvariableop[
Wsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_v_read_readvariableope
asavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_18_kernel_v_read_readvariableopc
_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_18_bias_v_read_readvariableope
asavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_19_kernel_v_read_readvariableopc
_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_19_bias_v_read_readvariableope
asavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_20_kernel_v_read_readvariableopc
_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_20_bias_v_read_readvariableope
asavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_21_kernel_v_read_readvariableopc
_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_21_bias_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_4_gamma_v_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_4_beta_v_read_readvariableopU
Qsavev2_adam_transformer_block_2_layer_normalization_5_gamma_v_read_readvariableopT
Psavev2_adam_transformer_block_2_layer_normalization_5_beta_v_read_readvariableop
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
ShardedFilename?1
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?0
value?0B?0dB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?
value?B?dB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?3
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_aux_output_kernel_read_readvariableop*savev2_aux_output_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop-savev2_main_output_kernel_read_readvariableop+savev2_main_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_4_embeddings_read_readvariableopPsavev2_token_and_position_embedding_2_embedding_5_embeddings_read_readvariableopZsavev2_transformer_block_2_multi_head_self_attention_2_dense_18_kernel_read_readvariableopXsavev2_transformer_block_2_multi_head_self_attention_2_dense_18_bias_read_readvariableopZsavev2_transformer_block_2_multi_head_self_attention_2_dense_19_kernel_read_readvariableopXsavev2_transformer_block_2_multi_head_self_attention_2_dense_19_bias_read_readvariableopZsavev2_transformer_block_2_multi_head_self_attention_2_dense_20_kernel_read_readvariableopXsavev2_transformer_block_2_multi_head_self_attention_2_dense_20_bias_read_readvariableopZsavev2_transformer_block_2_multi_head_self_attention_2_dense_21_kernel_read_readvariableopXsavev2_transformer_block_2_multi_head_self_attention_2_dense_21_bias_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableopJsavev2_transformer_block_2_layer_normalization_4_gamma_read_readvariableopIsavev2_transformer_block_2_layer_normalization_4_beta_read_readvariableopJsavev2_transformer_block_2_layer_normalization_5_gamma_read_readvariableopIsavev2_transformer_block_2_layer_normalization_5_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop3savev2_adam_aux_output_kernel_m_read_readvariableop1savev2_adam_aux_output_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop1savev2_adam_dense_25_kernel_m_read_readvariableop/savev2_adam_dense_25_bias_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop4savev2_adam_main_output_kernel_m_read_readvariableop2savev2_adam_main_output_bias_m_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_m_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_m_read_readvariableopasavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_18_kernel_m_read_readvariableop_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_18_bias_m_read_readvariableopasavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_19_kernel_m_read_readvariableop_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_19_bias_m_read_readvariableopasavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_20_kernel_m_read_readvariableop_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_20_bias_m_read_readvariableopasavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_21_kernel_m_read_readvariableop_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_21_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_4_gamma_m_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_4_beta_m_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_5_gamma_m_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_5_beta_m_read_readvariableop3savev2_adam_aux_output_kernel_v_read_readvariableop1savev2_adam_aux_output_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableop1savev2_adam_dense_25_kernel_v_read_readvariableop/savev2_adam_dense_25_bias_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop4savev2_adam_main_output_kernel_v_read_readvariableop2savev2_adam_main_output_bias_v_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_4_embeddings_v_read_readvariableopWsavev2_adam_token_and_position_embedding_2_embedding_5_embeddings_v_read_readvariableopasavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_18_kernel_v_read_readvariableop_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_18_bias_v_read_readvariableopasavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_19_kernel_v_read_readvariableop_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_19_bias_v_read_readvariableopasavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_20_kernel_v_read_readvariableop_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_20_bias_v_read_readvariableopasavev2_adam_transformer_block_2_multi_head_self_attention_2_dense_21_kernel_v_read_readvariableop_savev2_adam_transformer_block_2_multi_head_self_attention_2_dense_21_bias_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_4_gamma_v_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_4_beta_v_read_readvariableopQsavev2_adam_transformer_block_2_layer_normalization_5_gamma_v_read_readvariableopPsavev2_adam_transformer_block_2_layer_normalization_5_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: : ::@:@:@@:@:@@:@:@:: : : : : : :( :  : :  : :  : :  : :  : :  : : : : : : : : : : : : : : : : ::@:@:@@:@:@@:@:@:: :( :  : :  : :  : :  : :  : :  : : : : : : ::@:@:@@:@:@@:@:@:: :( :  : :  : :  : :  : :  : :  : : : : : : 2(
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

:@: 
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

:@: /
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

:@: K
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
??
?I
#__inference__traced_restore_1354443
file_prefix4
"assignvariableop_aux_output_kernel: 0
"assignvariableop_1_aux_output_bias:4
"assignvariableop_2_dense_24_kernel:@.
 assignvariableop_3_dense_24_bias:@4
"assignvariableop_4_dense_25_kernel:@@.
 assignvariableop_5_dense_25_bias:@4
"assignvariableop_6_dense_26_kernel:@@.
 assignvariableop_7_dense_26_bias:@7
%assignvariableop_8_main_output_kernel:@1
#assignvariableop_9_main_output_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: [
Iassignvariableop_15_token_and_position_embedding_2_embedding_4_embeddings: [
Iassignvariableop_16_token_and_position_embedding_2_embedding_5_embeddings:( e
Sassignvariableop_17_transformer_block_2_multi_head_self_attention_2_dense_18_kernel:  _
Qassignvariableop_18_transformer_block_2_multi_head_self_attention_2_dense_18_bias: e
Sassignvariableop_19_transformer_block_2_multi_head_self_attention_2_dense_19_kernel:  _
Qassignvariableop_20_transformer_block_2_multi_head_self_attention_2_dense_19_bias: e
Sassignvariableop_21_transformer_block_2_multi_head_self_attention_2_dense_20_kernel:  _
Qassignvariableop_22_transformer_block_2_multi_head_self_attention_2_dense_20_bias: e
Sassignvariableop_23_transformer_block_2_multi_head_self_attention_2_dense_21_kernel:  _
Qassignvariableop_24_transformer_block_2_multi_head_self_attention_2_dense_21_bias: 5
#assignvariableop_25_dense_22_kernel:  /
!assignvariableop_26_dense_22_bias: 5
#assignvariableop_27_dense_23_kernel:  /
!assignvariableop_28_dense_23_bias: Q
Cassignvariableop_29_transformer_block_2_layer_normalization_4_gamma: P
Bassignvariableop_30_transformer_block_2_layer_normalization_4_beta: Q
Cassignvariableop_31_transformer_block_2_layer_normalization_5_gamma: P
Bassignvariableop_32_transformer_block_2_layer_normalization_5_beta: #
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
*assignvariableop_44_adam_aux_output_bias_m:<
*assignvariableop_45_adam_dense_24_kernel_m:@6
(assignvariableop_46_adam_dense_24_bias_m:@<
*assignvariableop_47_adam_dense_25_kernel_m:@@6
(assignvariableop_48_adam_dense_25_bias_m:@<
*assignvariableop_49_adam_dense_26_kernel_m:@@6
(assignvariableop_50_adam_dense_26_bias_m:@?
-assignvariableop_51_adam_main_output_kernel_m:@9
+assignvariableop_52_adam_main_output_bias_m:b
Passignvariableop_53_adam_token_and_position_embedding_2_embedding_4_embeddings_m: b
Passignvariableop_54_adam_token_and_position_embedding_2_embedding_5_embeddings_m:( l
Zassignvariableop_55_adam_transformer_block_2_multi_head_self_attention_2_dense_18_kernel_m:  f
Xassignvariableop_56_adam_transformer_block_2_multi_head_self_attention_2_dense_18_bias_m: l
Zassignvariableop_57_adam_transformer_block_2_multi_head_self_attention_2_dense_19_kernel_m:  f
Xassignvariableop_58_adam_transformer_block_2_multi_head_self_attention_2_dense_19_bias_m: l
Zassignvariableop_59_adam_transformer_block_2_multi_head_self_attention_2_dense_20_kernel_m:  f
Xassignvariableop_60_adam_transformer_block_2_multi_head_self_attention_2_dense_20_bias_m: l
Zassignvariableop_61_adam_transformer_block_2_multi_head_self_attention_2_dense_21_kernel_m:  f
Xassignvariableop_62_adam_transformer_block_2_multi_head_self_attention_2_dense_21_bias_m: <
*assignvariableop_63_adam_dense_22_kernel_m:  6
(assignvariableop_64_adam_dense_22_bias_m: <
*assignvariableop_65_adam_dense_23_kernel_m:  6
(assignvariableop_66_adam_dense_23_bias_m: X
Jassignvariableop_67_adam_transformer_block_2_layer_normalization_4_gamma_m: W
Iassignvariableop_68_adam_transformer_block_2_layer_normalization_4_beta_m: X
Jassignvariableop_69_adam_transformer_block_2_layer_normalization_5_gamma_m: W
Iassignvariableop_70_adam_transformer_block_2_layer_normalization_5_beta_m: >
,assignvariableop_71_adam_aux_output_kernel_v: 8
*assignvariableop_72_adam_aux_output_bias_v:<
*assignvariableop_73_adam_dense_24_kernel_v:@6
(assignvariableop_74_adam_dense_24_bias_v:@<
*assignvariableop_75_adam_dense_25_kernel_v:@@6
(assignvariableop_76_adam_dense_25_bias_v:@<
*assignvariableop_77_adam_dense_26_kernel_v:@@6
(assignvariableop_78_adam_dense_26_bias_v:@?
-assignvariableop_79_adam_main_output_kernel_v:@9
+assignvariableop_80_adam_main_output_bias_v:b
Passignvariableop_81_adam_token_and_position_embedding_2_embedding_4_embeddings_v: b
Passignvariableop_82_adam_token_and_position_embedding_2_embedding_5_embeddings_v:( l
Zassignvariableop_83_adam_transformer_block_2_multi_head_self_attention_2_dense_18_kernel_v:  f
Xassignvariableop_84_adam_transformer_block_2_multi_head_self_attention_2_dense_18_bias_v: l
Zassignvariableop_85_adam_transformer_block_2_multi_head_self_attention_2_dense_19_kernel_v:  f
Xassignvariableop_86_adam_transformer_block_2_multi_head_self_attention_2_dense_19_bias_v: l
Zassignvariableop_87_adam_transformer_block_2_multi_head_self_attention_2_dense_20_kernel_v:  f
Xassignvariableop_88_adam_transformer_block_2_multi_head_self_attention_2_dense_20_bias_v: l
Zassignvariableop_89_adam_transformer_block_2_multi_head_self_attention_2_dense_21_kernel_v:  f
Xassignvariableop_90_adam_transformer_block_2_multi_head_self_attention_2_dense_21_bias_v: <
*assignvariableop_91_adam_dense_22_kernel_v:  6
(assignvariableop_92_adam_dense_22_bias_v: <
*assignvariableop_93_adam_dense_23_kernel_v:  6
(assignvariableop_94_adam_dense_23_bias_v: X
Jassignvariableop_95_adam_transformer_block_2_layer_normalization_4_gamma_v: W
Iassignvariableop_96_adam_transformer_block_2_layer_normalization_4_beta_v: X
Jassignvariableop_97_adam_transformer_block_2_layer_normalization_5_gamma_v: W
Iassignvariableop_98_adam_transformer_block_2_layer_normalization_5_beta_v: 
identity_100??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*?0
value?0B?0dB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_24_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_24_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_25_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_25_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_26_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_26_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOpIassignvariableop_15_token_and_position_embedding_2_embedding_4_embeddingsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpIassignvariableop_16_token_and_position_embedding_2_embedding_5_embeddingsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpSassignvariableop_17_transformer_block_2_multi_head_self_attention_2_dense_18_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpQassignvariableop_18_transformer_block_2_multi_head_self_attention_2_dense_18_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpSassignvariableop_19_transformer_block_2_multi_head_self_attention_2_dense_19_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpQassignvariableop_20_transformer_block_2_multi_head_self_attention_2_dense_19_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpSassignvariableop_21_transformer_block_2_multi_head_self_attention_2_dense_20_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpQassignvariableop_22_transformer_block_2_multi_head_self_attention_2_dense_20_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpSassignvariableop_23_transformer_block_2_multi_head_self_attention_2_dense_21_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpQassignvariableop_24_transformer_block_2_multi_head_self_attention_2_dense_21_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp#assignvariableop_25_dense_22_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp!assignvariableop_26_dense_22_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp#assignvariableop_27_dense_23_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp!assignvariableop_28_dense_23_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpCassignvariableop_29_transformer_block_2_layer_normalization_4_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpBassignvariableop_30_transformer_block_2_layer_normalization_4_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpCassignvariableop_31_transformer_block_2_layer_normalization_5_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpBassignvariableop_32_transformer_block_2_layer_normalization_5_betaIdentity_32:output:0"/device:CPU:0*
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
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_24_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_24_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_25_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_25_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_26_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_26_bias_mIdentity_50:output:0"/device:CPU:0*
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
AssignVariableOp_53AssignVariableOpPassignvariableop_53_adam_token_and_position_embedding_2_embedding_4_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpPassignvariableop_54_adam_token_and_position_embedding_2_embedding_5_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpZassignvariableop_55_adam_transformer_block_2_multi_head_self_attention_2_dense_18_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpXassignvariableop_56_adam_transformer_block_2_multi_head_self_attention_2_dense_18_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpZassignvariableop_57_adam_transformer_block_2_multi_head_self_attention_2_dense_19_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpXassignvariableop_58_adam_transformer_block_2_multi_head_self_attention_2_dense_19_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpZassignvariableop_59_adam_transformer_block_2_multi_head_self_attention_2_dense_20_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpXassignvariableop_60_adam_transformer_block_2_multi_head_self_attention_2_dense_20_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpZassignvariableop_61_adam_transformer_block_2_multi_head_self_attention_2_dense_21_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpXassignvariableop_62_adam_transformer_block_2_multi_head_self_attention_2_dense_21_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_22_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_22_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_23_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_23_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpJassignvariableop_67_adam_transformer_block_2_layer_normalization_4_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpIassignvariableop_68_adam_transformer_block_2_layer_normalization_4_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpJassignvariableop_69_adam_transformer_block_2_layer_normalization_5_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOpIassignvariableop_70_adam_transformer_block_2_layer_normalization_5_beta_mIdentity_70:output:0"/device:CPU:0*
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
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_24_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_24_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_dense_25_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_dense_25_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_26_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_26_bias_vIdentity_78:output:0"/device:CPU:0*
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
AssignVariableOp_81AssignVariableOpPassignvariableop_81_adam_token_and_position_embedding_2_embedding_4_embeddings_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOpPassignvariableop_82_adam_token_and_position_embedding_2_embedding_5_embeddings_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOpZassignvariableop_83_adam_transformer_block_2_multi_head_self_attention_2_dense_18_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOpXassignvariableop_84_adam_transformer_block_2_multi_head_self_attention_2_dense_18_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOpZassignvariableop_85_adam_transformer_block_2_multi_head_self_attention_2_dense_19_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOpXassignvariableop_86_adam_transformer_block_2_multi_head_self_attention_2_dense_19_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOpZassignvariableop_87_adam_transformer_block_2_multi_head_self_attention_2_dense_20_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOpXassignvariableop_88_adam_transformer_block_2_multi_head_self_attention_2_dense_20_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOpZassignvariableop_89_adam_transformer_block_2_multi_head_self_attention_2_dense_21_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOpXassignvariableop_90_adam_transformer_block_2_multi_head_self_attention_2_dense_21_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_dense_22_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_dense_22_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_dense_23_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_dense_23_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOpJassignvariableop_95_adam_transformer_block_2_layer_normalization_4_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOpIassignvariableop_96_adam_transformer_block_2_layer_normalization_4_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOpJassignvariableop_97_adam_transformer_block_2_layer_normalization_5_gamma_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOpIassignvariableop_98_adam_transformer_block_2_layer_normalization_5_beta_vIdentity_98:output:0"/device:CPU:0*
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
?
E__inference_dense_24_layer_call_and_return_conditional_losses_1351097

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
E__inference_dense_22_layer_call_and_return_conditional_losses_1350557

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
?
?
5__inference_transformer_block_2_layer_call_fn_1352958

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
GPU 2J 8? *Y
fTRR
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_13515772
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
?
v
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1353515
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
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
inputs/1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
	aux_input2
serving_default_aux_input:0?????????
;
input_30
serving_default_input_3:0?????????(>

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
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
?
	token_emb
pos_emb
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
regularization_losses
	variables
 trainable_variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"regularization_losses
#	variables
$trainable_variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'bias
(regularization_losses
)	variables
*trainable_variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
?
,regularization_losses
-	variables
.trainable_variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

0kernel
1bias
2regularization_losses
3	variables
4trainable_variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

6kernel
7bias
8regularization_losses
9	variables
:trainable_variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Bkernel
Cbias
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Hiter

Ibeta_1

Jbeta_2
	Kdecay
Llearning_rate&m?'m?0m?1m?6m?7m?<m?=m?Bm?Cm?Mm?Nm?Om?Pm?Qm?Rm?Sm?Tm?Um?Vm?Wm?Xm?Ym?Zm?[m?\m?]m?^m?&v?'v?0v?1v?6v?7v?<v?=v?Bv?Cv?Mv?Nv?Ov?Pv?Qv?Rv?Sv?Tv?Uv?Vv?Wv?Xv?Yv?Zv?[v?\v?]v?^v?"
	optimizer
 "
trackable_list_wrapper
?
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
&18
'19
020
121
622
723
<24
=25
B26
C27"
trackable_list_wrapper
?
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
&18
'19
020
121
622
723
<24
=25
B26
C27"
trackable_list_wrapper
?
_metrics
`layer_metrics
regularization_losses
alayer_regularization_losses
	variables
bnon_trainable_variables

clayers
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
M
embeddings
dregularization_losses
e	variables
ftrainable_variables
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
N
embeddings
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
?
lmetrics
mlayer_metrics
regularization_losses
nlayer_regularization_losses
	variables
onon_trainable_variables

players
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
qquery_dense
r	key_dense
svalue_dense
tcombine_heads
uregularization_losses
v	variables
wtrainable_variables
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
ylayer_with_weights-0
ylayer-0
zlayer_with_weights-1
zlayer-1
{regularization_losses
|	variables
}trainable_variables
~	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
axis
	[gamma
\beta
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	]gamma
^beta
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
?
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15"
trackable_list_wrapper
?
O0
P1
Q2
R3
S4
T5
U6
V7
W8
X9
Y10
Z11
[12
\13
]14
^15"
trackable_list_wrapper
?
?metrics
?layer_metrics
regularization_losses
 ?layer_regularization_losses
	variables
?non_trainable_variables
?layers
 trainable_variables
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
?metrics
?layer_metrics
"regularization_losses
 ?layer_regularization_losses
#	variables
?non_trainable_variables
?layers
$trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:! 2aux_output/kernel
:2aux_output/bias
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
?metrics
?layer_metrics
(regularization_losses
 ?layer_regularization_losses
)	variables
?non_trainable_variables
?layers
*trainable_variables
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
?metrics
?layer_metrics
,regularization_losses
 ?layer_regularization_losses
-	variables
?non_trainable_variables
?layers
.trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_24/kernel
:@2dense_24/bias
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
?metrics
?layer_metrics
2regularization_losses
 ?layer_regularization_losses
3	variables
?non_trainable_variables
?layers
4trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_25/kernel
:@2dense_25/bias
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
?metrics
?layer_metrics
8regularization_losses
 ?layer_regularization_losses
9	variables
?non_trainable_variables
?layers
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_26/kernel
:@2dense_26/bias
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
?metrics
?layer_metrics
>regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
@trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"@2main_output/kernel
:2main_output/bias
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
?metrics
?layer_metrics
Dregularization_losses
 ?layer_regularization_losses
E	variables
?non_trainable_variables
?layers
Ftrainable_variables
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
G:E 25token_and_position_embedding_2/embedding_4/embeddings
G:E( 25token_and_position_embedding_2/embedding_5/embeddings
Q:O  2?transformer_block_2/multi_head_self_attention_2/dense_18/kernel
K:I 2=transformer_block_2/multi_head_self_attention_2/dense_18/bias
Q:O  2?transformer_block_2/multi_head_self_attention_2/dense_19/kernel
K:I 2=transformer_block_2/multi_head_self_attention_2/dense_19/bias
Q:O  2?transformer_block_2/multi_head_self_attention_2/dense_20/kernel
K:I 2=transformer_block_2/multi_head_self_attention_2/dense_20/bias
Q:O  2?transformer_block_2/multi_head_self_attention_2/dense_21/kernel
K:I 2=transformer_block_2/multi_head_self_attention_2/dense_21/bias
!:  2dense_22/kernel
: 2dense_22/bias
!:  2dense_23/kernel
: 2dense_23/bias
=:; 2/transformer_block_2/layer_normalization_4/gamma
<:: 2.transformer_block_2/layer_normalization_4/beta
=:; 2/transformer_block_2/layer_normalization_5/gamma
<:: 2.transformer_block_2/layer_normalization_5/beta
H
?0
?1
?2
?3
?4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
n
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
10"
trackable_list_wrapper
 "
trackable_list_wrapper
'
M0"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
?
?metrics
?layer_metrics
dregularization_losses
 ?layer_regularization_losses
e	variables
?non_trainable_variables
?layers
ftrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
?
?metrics
?layer_metrics
hregularization_losses
 ?layer_regularization_losses
i	variables
?non_trainable_variables
?layers
jtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Okernel
Pbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Qkernel
Rbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Skernel
Tbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ukernel
Vbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
X
O0
P1
Q2
R3
S4
T5
U6
V7"
trackable_list_wrapper
X
O0
P1
Q2
R3
S4
T5
U6
V7"
trackable_list_wrapper
?
?metrics
?layer_metrics
uregularization_losses
 ?layer_regularization_losses
v	variables
?non_trainable_variables
?layers
wtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Wkernel
Xbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ykernel
Zbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
<
W0
X1
Y2
Z3"
trackable_list_wrapper
<
W0
X1
Y2
Z3"
trackable_list_wrapper
?
?metrics
?layer_metrics
{regularization_losses
 ?layer_regularization_losses
|	variables
?non_trainable_variables
?layers
}trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
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
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
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
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
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
R

?total

?count
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
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
q0
r1
s2
t3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
?
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?	variables
?non_trainable_variables
?layers
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
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
:  (2total
:  (2count
0
?0
?1"
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
(:& 2Adam/aux_output/kernel/m
": 2Adam/aux_output/bias/m
&:$@2Adam/dense_24/kernel/m
 :@2Adam/dense_24/bias/m
&:$@@2Adam/dense_25/kernel/m
 :@2Adam/dense_25/bias/m
&:$@@2Adam/dense_26/kernel/m
 :@2Adam/dense_26/bias/m
):'@2Adam/main_output/kernel/m
#:!2Adam/main_output/bias/m
L:J 2<Adam/token_and_position_embedding_2/embedding_4/embeddings/m
L:J( 2<Adam/token_and_position_embedding_2/embedding_5/embeddings/m
V:T  2FAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/m
P:N 2DAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/m
V:T  2FAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/m
P:N 2DAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/m
V:T  2FAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/m
P:N 2DAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/m
V:T  2FAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/m
P:N 2DAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/m
&:$  2Adam/dense_22/kernel/m
 : 2Adam/dense_22/bias/m
&:$  2Adam/dense_23/kernel/m
 : 2Adam/dense_23/bias/m
B:@ 26Adam/transformer_block_2/layer_normalization_4/gamma/m
A:? 25Adam/transformer_block_2/layer_normalization_4/beta/m
B:@ 26Adam/transformer_block_2/layer_normalization_5/gamma/m
A:? 25Adam/transformer_block_2/layer_normalization_5/beta/m
(:& 2Adam/aux_output/kernel/v
": 2Adam/aux_output/bias/v
&:$@2Adam/dense_24/kernel/v
 :@2Adam/dense_24/bias/v
&:$@@2Adam/dense_25/kernel/v
 :@2Adam/dense_25/bias/v
&:$@@2Adam/dense_26/kernel/v
 :@2Adam/dense_26/bias/v
):'@2Adam/main_output/kernel/v
#:!2Adam/main_output/bias/v
L:J 2<Adam/token_and_position_embedding_2/embedding_4/embeddings/v
L:J( 2<Adam/token_and_position_embedding_2/embedding_5/embeddings/v
V:T  2FAdam/transformer_block_2/multi_head_self_attention_2/dense_18/kernel/v
P:N 2DAdam/transformer_block_2/multi_head_self_attention_2/dense_18/bias/v
V:T  2FAdam/transformer_block_2/multi_head_self_attention_2/dense_19/kernel/v
P:N 2DAdam/transformer_block_2/multi_head_self_attention_2/dense_19/bias/v
V:T  2FAdam/transformer_block_2/multi_head_self_attention_2/dense_20/kernel/v
P:N 2DAdam/transformer_block_2/multi_head_self_attention_2/dense_20/bias/v
V:T  2FAdam/transformer_block_2/multi_head_self_attention_2/dense_21/kernel/v
P:N 2DAdam/transformer_block_2/multi_head_self_attention_2/dense_21/bias/v
&:$  2Adam/dense_22/kernel/v
 : 2Adam/dense_22/bias/v
&:$  2Adam/dense_23/kernel/v
 : 2Adam/dense_23/bias/v
B:@ 26Adam/transformer_block_2/layer_normalization_4/gamma/v
A:? 25Adam/transformer_block_2/layer_normalization_4/beta/v
B:@ 26Adam/transformer_block_2/layer_normalization_5/gamma/v
A:? 25Adam/transformer_block_2/layer_normalization_5/beta/v
?2?
)__inference_model_2_layer_call_fn_1351217
)__inference_model_2_layer_call_fn_1352163
)__inference_model_2_layer_call_fn_1352227
)__inference_model_2_layer_call_fn_1351885?
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
"__inference__wrapped_model_1350519input_3	aux_input"?
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
?2?
D__inference_model_2_layer_call_and_return_conditional_losses_1352532
D__inference_model_2_layer_call_and_return_conditional_losses_1352851
D__inference_model_2_layer_call_and_return_conditional_losses_1351956
D__inference_model_2_layer_call_and_return_conditional_losses_1352027?
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
?2?
@__inference_token_and_position_embedding_2_layer_call_fn_1352860?
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
?2?
[__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_1352884?
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
5__inference_transformer_block_2_layer_call_fn_1352921
5__inference_transformer_block_2_layer_call_fn_1352958?
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
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_1353202
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_1353460?
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
<__inference_global_average_pooling1d_2_layer_call_fn_1353465
<__inference_global_average_pooling1d_2_layer_call_fn_1353470?
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
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1353476
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1353482?
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
,__inference_aux_output_layer_call_fn_1353491?
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
G__inference_aux_output_layer_call_and_return_conditional_losses_1353502?
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
/__inference_concatenate_2_layer_call_fn_1353508?
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
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1353515?
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
*__inference_dense_24_layer_call_fn_1353524?
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
E__inference_dense_24_layer_call_and_return_conditional_losses_1353535?
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
*__inference_dense_25_layer_call_fn_1353544?
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
E__inference_dense_25_layer_call_and_return_conditional_losses_1353555?
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
*__inference_dense_26_layer_call_fn_1353564?
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
E__inference_dense_26_layer_call_and_return_conditional_losses_1353575?
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
-__inference_main_output_layer_call_fn_1353584?
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
H__inference_main_output_layer_call_and_return_conditional_losses_1353595?
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
%__inference_signature_wrapper_1352099	aux_inputinput_3"?
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
.__inference_sequential_2_layer_call_fn_1350611
.__inference_sequential_2_layer_call_fn_1353608
.__inference_sequential_2_layer_call_fn_1353621
.__inference_sequential_2_layer_call_fn_1350684?
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
I__inference_sequential_2_layer_call_and_return_conditional_losses_1353678
I__inference_sequential_2_layer_call_and_return_conditional_losses_1353735
I__inference_sequential_2_layer_call_and_return_conditional_losses_1350698
I__inference_sequential_2_layer_call_and_return_conditional_losses_1350712?
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
*__inference_dense_22_layer_call_fn_1353744?
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
E__inference_dense_22_layer_call_and_return_conditional_losses_1353775?
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
*__inference_dense_23_layer_call_fn_1353784?
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
E__inference_dense_23_layer_call_and_return_conditional_losses_1353814?
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
"__inference__wrapped_model_1350519?NMOPQRSTUV[\WXYZ]^&'0167<=BCZ?W
P?M
K?H
!?
input_3?????????(
#? 
	aux_input?????????
? "m?j
2

aux_output$?!

aux_output?????????
4
main_output%?"
main_output??????????
G__inference_aux_output_layer_call_and_return_conditional_losses_1353502\&'/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? 
,__inference_aux_output_layer_call_fn_1353491O&'/?,
%?"
 ?
inputs????????? 
? "???????????
J__inference_concatenate_2_layer_call_and_return_conditional_losses_1353515?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
/__inference_concatenate_2_layer_call_fn_1353508vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
E__inference_dense_22_layer_call_and_return_conditional_losses_1353775dWX3?0
)?&
$?!
inputs?????????( 
? ")?&
?
0?????????( 
? ?
*__inference_dense_22_layer_call_fn_1353744WWX3?0
)?&
$?!
inputs?????????( 
? "??????????( ?
E__inference_dense_23_layer_call_and_return_conditional_losses_1353814dYZ3?0
)?&
$?!
inputs?????????( 
? ")?&
?
0?????????( 
? ?
*__inference_dense_23_layer_call_fn_1353784WYZ3?0
)?&
$?!
inputs?????????( 
? "??????????( ?
E__inference_dense_24_layer_call_and_return_conditional_losses_1353535\01/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? }
*__inference_dense_24_layer_call_fn_1353524O01/?,
%?"
 ?
inputs?????????
? "??????????@?
E__inference_dense_25_layer_call_and_return_conditional_losses_1353555\67/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? }
*__inference_dense_25_layer_call_fn_1353544O67/?,
%?"
 ?
inputs?????????@
? "??????????@?
E__inference_dense_26_layer_call_and_return_conditional_losses_1353575\<=/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? }
*__inference_dense_26_layer_call_fn_1353564O<=/?,
%?"
 ?
inputs?????????@
? "??????????@?
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1353476{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
W__inference_global_average_pooling1d_2_layer_call_and_return_conditional_losses_1353482`7?4
-?*
$?!
inputs?????????( 

 
? "%?"
?
0????????? 
? ?
<__inference_global_average_pooling1d_2_layer_call_fn_1353465nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
<__inference_global_average_pooling1d_2_layer_call_fn_1353470S7?4
-?*
$?!
inputs?????????( 

 
? "?????????? ?
H__inference_main_output_layer_call_and_return_conditional_losses_1353595\BC/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ?
-__inference_main_output_layer_call_fn_1353584OBC/?,
%?"
 ?
inputs?????????@
? "???????????
D__inference_model_2_layer_call_and_return_conditional_losses_1351956?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
!?
input_3?????????(
#? 
	aux_input?????????
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
D__inference_model_2_layer_call_and_return_conditional_losses_1352027?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
!?
input_3?????????(
#? 
	aux_input?????????
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
D__inference_model_2_layer_call_and_return_conditional_losses_1352532?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
"?
inputs/0?????????(
"?
inputs/1?????????
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
D__inference_model_2_layer_call_and_return_conditional_losses_1352851?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
"?
inputs/0?????????(
"?
inputs/1?????????
p

 
? "K?H
A?>
?
0/0?????????
?
0/1?????????
? ?
)__inference_model_2_layer_call_fn_1351217?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
!?
input_3?????????(
#? 
	aux_input?????????
p 

 
? "=?:
?
0?????????
?
1??????????
)__inference_model_2_layer_call_fn_1351885?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
!?
input_3?????????(
#? 
	aux_input?????????
p

 
? "=?:
?
0?????????
?
1??????????
)__inference_model_2_layer_call_fn_1352163?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
"?
inputs/0?????????(
"?
inputs/1?????????
p 

 
? "=?:
?
0?????????
?
1??????????
)__inference_model_2_layer_call_fn_1352227?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
"?
inputs/0?????????(
"?
inputs/1?????????
p

 
? "=?:
?
0?????????
?
1??????????
I__inference_sequential_2_layer_call_and_return_conditional_losses_1350698vWXYZC?@
9?6
,?)
dense_22_input?????????( 
p 

 
? ")?&
?
0?????????( 
? ?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1350712vWXYZC?@
9?6
,?)
dense_22_input?????????( 
p

 
? ")?&
?
0?????????( 
? ?
I__inference_sequential_2_layer_call_and_return_conditional_losses_1353678nWXYZ;?8
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
I__inference_sequential_2_layer_call_and_return_conditional_losses_1353735nWXYZ;?8
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
.__inference_sequential_2_layer_call_fn_1350611iWXYZC?@
9?6
,?)
dense_22_input?????????( 
p 

 
? "??????????( ?
.__inference_sequential_2_layer_call_fn_1350684iWXYZC?@
9?6
,?)
dense_22_input?????????( 
p

 
? "??????????( ?
.__inference_sequential_2_layer_call_fn_1353608aWXYZ;?8
1?.
$?!
inputs?????????( 
p 

 
? "??????????( ?
.__inference_sequential_2_layer_call_fn_1353621aWXYZ;?8
1?.
$?!
inputs?????????( 
p

 
? "??????????( ?
%__inference_signature_wrapper_1352099?NMOPQRSTUV[\WXYZ]^&'0167<=BCm?j
? 
c?`
0
	aux_input#? 
	aux_input?????????
,
input_3!?
input_3?????????("m?j
2

aux_output$?!

aux_output?????????
4
main_output%?"
main_output??????????
[__inference_token_and_position_embedding_2_layer_call_and_return_conditional_losses_1352884[NM*?'
 ?
?
x?????????(
? ")?&
?
0?????????( 
? ?
@__inference_token_and_position_embedding_2_layer_call_fn_1352860NNM*?'
 ?
?
x?????????(
? "??????????( ?
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_1353202vOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????( 
p 
? ")?&
?
0?????????( 
? ?
P__inference_transformer_block_2_layer_call_and_return_conditional_losses_1353460vOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????( 
p
? ")?&
?
0?????????( 
? ?
5__inference_transformer_block_2_layer_call_fn_1352921iOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????( 
p 
? "??????????( ?
5__inference_transformer_block_2_layer_call_fn_1352958iOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????( 
p
? "??????????( 