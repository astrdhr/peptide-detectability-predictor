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
dense_78/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_78/kernel
s
#dense_78/kernel/Read/ReadVariableOpReadVariableOpdense_78/kernel*
_output_shapes

:@*
dtype0
r
dense_78/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_78/bias
k
!dense_78/bias/Read/ReadVariableOpReadVariableOpdense_78/bias*
_output_shapes
:@*
dtype0
z
dense_79/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_79/kernel
s
#dense_79/kernel/Read/ReadVariableOpReadVariableOpdense_79/kernel*
_output_shapes

:@@*
dtype0
r
dense_79/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_79/bias
k
!dense_79/bias/Read/ReadVariableOpReadVariableOpdense_79/bias*
_output_shapes
:@*
dtype0
z
dense_80/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_80/kernel
s
#dense_80/kernel/Read/ReadVariableOpReadVariableOpdense_80/kernel*
_output_shapes

:@@*
dtype0
r
dense_80/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_80/bias
k
!dense_80/bias/Read/ReadVariableOpReadVariableOpdense_80/bias*
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
6token_and_position_embedding_8/embedding_16/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *G
shared_name86token_and_position_embedding_8/embedding_16/embeddings
?
Jtoken_and_position_embedding_8/embedding_16/embeddings/Read/ReadVariableOpReadVariableOp6token_and_position_embedding_8/embedding_16/embeddings*
_output_shapes

: *
dtype0
?
6token_and_position_embedding_8/embedding_17/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *G
shared_name86token_and_position_embedding_8/embedding_17/embeddings
?
Jtoken_and_position_embedding_8/embedding_17/embeddings/Read/ReadVariableOpReadVariableOp6token_and_position_embedding_8/embedding_17/embeddings*
_output_shapes

:( *
dtype0
?
?transformer_block_8/multi_head_self_attention_8/dense_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *P
shared_nameA?transformer_block_8/multi_head_self_attention_8/dense_72/kernel
?
Stransformer_block_8/multi_head_self_attention_8/dense_72/kernel/Read/ReadVariableOpReadVariableOp?transformer_block_8/multi_head_self_attention_8/dense_72/kernel*
_output_shapes

:  *
dtype0
?
=transformer_block_8/multi_head_self_attention_8/dense_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=transformer_block_8/multi_head_self_attention_8/dense_72/bias
?
Qtransformer_block_8/multi_head_self_attention_8/dense_72/bias/Read/ReadVariableOpReadVariableOp=transformer_block_8/multi_head_self_attention_8/dense_72/bias*
_output_shapes
: *
dtype0
?
?transformer_block_8/multi_head_self_attention_8/dense_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *P
shared_nameA?transformer_block_8/multi_head_self_attention_8/dense_73/kernel
?
Stransformer_block_8/multi_head_self_attention_8/dense_73/kernel/Read/ReadVariableOpReadVariableOp?transformer_block_8/multi_head_self_attention_8/dense_73/kernel*
_output_shapes

:  *
dtype0
?
=transformer_block_8/multi_head_self_attention_8/dense_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=transformer_block_8/multi_head_self_attention_8/dense_73/bias
?
Qtransformer_block_8/multi_head_self_attention_8/dense_73/bias/Read/ReadVariableOpReadVariableOp=transformer_block_8/multi_head_self_attention_8/dense_73/bias*
_output_shapes
: *
dtype0
?
?transformer_block_8/multi_head_self_attention_8/dense_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *P
shared_nameA?transformer_block_8/multi_head_self_attention_8/dense_74/kernel
?
Stransformer_block_8/multi_head_self_attention_8/dense_74/kernel/Read/ReadVariableOpReadVariableOp?transformer_block_8/multi_head_self_attention_8/dense_74/kernel*
_output_shapes

:  *
dtype0
?
=transformer_block_8/multi_head_self_attention_8/dense_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=transformer_block_8/multi_head_self_attention_8/dense_74/bias
?
Qtransformer_block_8/multi_head_self_attention_8/dense_74/bias/Read/ReadVariableOpReadVariableOp=transformer_block_8/multi_head_self_attention_8/dense_74/bias*
_output_shapes
: *
dtype0
?
?transformer_block_8/multi_head_self_attention_8/dense_75/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *P
shared_nameA?transformer_block_8/multi_head_self_attention_8/dense_75/kernel
?
Stransformer_block_8/multi_head_self_attention_8/dense_75/kernel/Read/ReadVariableOpReadVariableOp?transformer_block_8/multi_head_self_attention_8/dense_75/kernel*
_output_shapes

:  *
dtype0
?
=transformer_block_8/multi_head_self_attention_8/dense_75/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *N
shared_name?=transformer_block_8/multi_head_self_attention_8/dense_75/bias
?
Qtransformer_block_8/multi_head_self_attention_8/dense_75/bias/Read/ReadVariableOpReadVariableOp=transformer_block_8/multi_head_self_attention_8/dense_75/bias*
_output_shapes
: *
dtype0
z
dense_76/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_76/kernel
s
#dense_76/kernel/Read/ReadVariableOpReadVariableOpdense_76/kernel*
_output_shapes

:  *
dtype0
r
dense_76/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_76/bias
k
!dense_76/bias/Read/ReadVariableOpReadVariableOpdense_76/bias*
_output_shapes
: *
dtype0
z
dense_77/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_77/kernel
s
#dense_77/kernel/Read/ReadVariableOpReadVariableOpdense_77/kernel*
_output_shapes

:  *
dtype0
r
dense_77/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_77/bias
k
!dense_77/bias/Read/ReadVariableOpReadVariableOpdense_77/bias*
_output_shapes
: *
dtype0
?
0transformer_block_8/layer_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_8/layer_normalization_16/gamma
?
Dtransformer_block_8/layer_normalization_16/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_8/layer_normalization_16/gamma*
_output_shapes
: *
dtype0
?
/transformer_block_8/layer_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_8/layer_normalization_16/beta
?
Ctransformer_block_8/layer_normalization_16/beta/Read/ReadVariableOpReadVariableOp/transformer_block_8/layer_normalization_16/beta*
_output_shapes
: *
dtype0
?
0transformer_block_8/layer_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20transformer_block_8/layer_normalization_17/gamma
?
Dtransformer_block_8/layer_normalization_17/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_8/layer_normalization_17/gamma*
_output_shapes
: *
dtype0
?
/transformer_block_8/layer_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/transformer_block_8/layer_normalization_17/beta
?
Ctransformer_block_8/layer_normalization_17/beta/Read/ReadVariableOpReadVariableOp/transformer_block_8/layer_normalization_17/beta*
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
Adam/dense_78/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_78/kernel/m
?
*Adam/dense_78/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_78/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_78/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_78/bias/m
y
(Adam/dense_78/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_78/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_79/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_79/kernel/m
?
*Adam/dense_79/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_79/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_79/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_79/bias/m
y
(Adam/dense_79/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_79/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_80/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_80/kernel/m
?
*Adam/dense_80/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_80/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_80/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_80/bias/m
y
(Adam/dense_80/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_80/bias/m*
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
=Adam/token_and_position_embedding_8/embedding_16/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *N
shared_name?=Adam/token_and_position_embedding_8/embedding_16/embeddings/m
?
QAdam/token_and_position_embedding_8/embedding_16/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_8/embedding_16/embeddings/m*
_output_shapes

: *
dtype0
?
=Adam/token_and_position_embedding_8/embedding_17/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *N
shared_name?=Adam/token_and_position_embedding_8/embedding_17/embeddings/m
?
QAdam/token_and_position_embedding_8/embedding_17/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_8/embedding_17/embeddings/m*
_output_shapes

:( *
dtype0
?
FAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/m
?
ZAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/m*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/m
?
XAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/m/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/m*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/m
?
ZAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/m*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/m
?
XAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/m/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/m*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/m
?
ZAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/m*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/m
?
XAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/m/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/m*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/m
?
ZAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/m/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/m*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/m
?
XAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/m/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_76/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_76/kernel/m
?
*Adam/dense_76/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_76/kernel/m*
_output_shapes

:  *
dtype0
?
Adam/dense_76/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_76/bias/m
y
(Adam/dense_76/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_76/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_77/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_77/kernel/m
?
*Adam/dense_77/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_77/kernel/m*
_output_shapes

:  *
dtype0
?
Adam/dense_77/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_77/bias/m
y
(Adam/dense_77/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_77/bias/m*
_output_shapes
: *
dtype0
?
7Adam/transformer_block_8/layer_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_8/layer_normalization_16/gamma/m
?
KAdam/transformer_block_8/layer_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_16/gamma/m*
_output_shapes
: *
dtype0
?
6Adam/transformer_block_8/layer_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_8/layer_normalization_16/beta/m
?
JAdam/transformer_block_8/layer_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_16/beta/m*
_output_shapes
: *
dtype0
?
7Adam/transformer_block_8/layer_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_8/layer_normalization_17/gamma/m
?
KAdam/transformer_block_8/layer_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_17/gamma/m*
_output_shapes
: *
dtype0
?
6Adam/transformer_block_8/layer_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_8/layer_normalization_17/beta/m
?
JAdam/transformer_block_8/layer_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_17/beta/m*
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
Adam/dense_78/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_78/kernel/v
?
*Adam/dense_78/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_78/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_78/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_78/bias/v
y
(Adam/dense_78/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_78/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_79/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_79/kernel/v
?
*Adam/dense_79/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_79/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_79/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_79/bias/v
y
(Adam/dense_79/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_79/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_80/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_80/kernel/v
?
*Adam/dense_80/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_80/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_80/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_80/bias/v
y
(Adam/dense_80/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_80/bias/v*
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
=Adam/token_and_position_embedding_8/embedding_16/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *N
shared_name?=Adam/token_and_position_embedding_8/embedding_16/embeddings/v
?
QAdam/token_and_position_embedding_8/embedding_16/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_8/embedding_16/embeddings/v*
_output_shapes

: *
dtype0
?
=Adam/token_and_position_embedding_8/embedding_17/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *N
shared_name?=Adam/token_and_position_embedding_8/embedding_17/embeddings/v
?
QAdam/token_and_position_embedding_8/embedding_17/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_8/embedding_17/embeddings/v*
_output_shapes

:( *
dtype0
?
FAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/v
?
ZAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/v*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/v
?
XAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/v/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/v*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/v
?
ZAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/v*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/v
?
XAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/v/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/v*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/v
?
ZAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/v*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/v
?
XAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/v/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/v*
_output_shapes
: *
dtype0
?
FAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *W
shared_nameHFAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/v
?
ZAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/v/Read/ReadVariableOpReadVariableOpFAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/v*
_output_shapes

:  *
dtype0
?
DAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *U
shared_nameFDAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/v
?
XAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/v/Read/ReadVariableOpReadVariableOpDAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_76/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_76/kernel/v
?
*Adam/dense_76/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_76/kernel/v*
_output_shapes

:  *
dtype0
?
Adam/dense_76/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_76/bias/v
y
(Adam/dense_76/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_76/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_77/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *'
shared_nameAdam/dense_77/kernel/v
?
*Adam/dense_77/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_77/kernel/v*
_output_shapes

:  *
dtype0
?
Adam/dense_77/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_77/bias/v
y
(Adam/dense_77/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_77/bias/v*
_output_shapes
: *
dtype0
?
7Adam/transformer_block_8/layer_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_8/layer_normalization_16/gamma/v
?
KAdam/transformer_block_8/layer_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_16/gamma/v*
_output_shapes
: *
dtype0
?
6Adam/transformer_block_8/layer_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_8/layer_normalization_16/beta/v
?
JAdam/transformer_block_8/layer_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_16/beta/v*
_output_shapes
: *
dtype0
?
7Adam/transformer_block_8/layer_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/transformer_block_8/layer_normalization_17/gamma/v
?
KAdam/transformer_block_8/layer_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_17/gamma/v*
_output_shapes
: *
dtype0
?
6Adam/transformer_block_8/layer_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86Adam/transformer_block_8/layer_normalization_17/beta/v
?
JAdam/transformer_block_8/layer_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_17/beta/v*
_output_shapes
: *
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ڲ
valueϲB˲ Bò
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
trainable_variables
	variables
	keras_api

signatures
 
n
	token_emb
pos_emb
regularization_losses
trainable_variables
	variables
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
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
h

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
 
R
,regularization_losses
-trainable_variables
.	variables
/	keras_api
h

0kernel
1bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
h

<kernel
=bias
>regularization_losses
?trainable_variables
@	variables
A	keras_api
h

Bkernel
Cbias
Dregularization_losses
Etrainable_variables
F	variables
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
regularization_losses
_non_trainable_variables
trainable_variables
`layer_metrics
	variables
ametrics

blayers
clayer_regularization_losses
 
b
M
embeddings
dregularization_losses
etrainable_variables
f	variables
g	keras_api
b
N
embeddings
hregularization_losses
itrainable_variables
j	variables
k	keras_api
 

M0
N1

M0
N1
?
regularization_losses
lnon_trainable_variables
trainable_variables
mlayer_metrics
	variables
nmetrics

olayers
player_regularization_losses
?
qquery_dense
r	key_dense
svalue_dense
tcombine_heads
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
?
ylayer_with_weights-0
ylayer-0
zlayer_with_weights-1
zlayer-1
{regularization_losses
|trainable_variables
}	variables
~	keras_api
u
axis
	[gamma
\beta
?regularization_losses
?trainable_variables
?	variables
?	keras_api
v
	?axis
	]gamma
^beta
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
regularization_losses
?non_trainable_variables
trainable_variables
?layer_metrics
 	variables
?metrics
?layers
 ?layer_regularization_losses
 
 
 
?
"regularization_losses
?non_trainable_variables
#trainable_variables
?layer_metrics
$	variables
?metrics
?layers
 ?layer_regularization_losses
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
(regularization_losses
?non_trainable_variables
)trainable_variables
?layer_metrics
*	variables
?metrics
?layers
 ?layer_regularization_losses
 
 
 
?
,regularization_losses
?non_trainable_variables
-trainable_variables
?layer_metrics
.	variables
?metrics
?layers
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_78/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_78/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

00
11

00
11
?
2regularization_losses
?non_trainable_variables
3trainable_variables
?layer_metrics
4	variables
?metrics
?layers
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_79/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_79/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
?
8regularization_losses
?non_trainable_variables
9trainable_variables
?layer_metrics
:	variables
?metrics
?layers
 ?layer_regularization_losses
[Y
VARIABLE_VALUEdense_80/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_80/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
?
>regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
@	variables
?metrics
?layers
 ?layer_regularization_losses
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
Dregularization_losses
?non_trainable_variables
Etrainable_variables
?layer_metrics
F	variables
?metrics
?layers
 ?layer_regularization_losses
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
|z
VARIABLE_VALUE6token_and_position_embedding_8/embedding_16/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE6token_and_position_embedding_8/embedding_17/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?transformer_block_8/multi_head_self_attention_8/dense_72/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=transformer_block_8/multi_head_self_attention_8/dense_72/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?transformer_block_8/multi_head_self_attention_8/dense_73/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=transformer_block_8/multi_head_self_attention_8/dense_73/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?transformer_block_8/multi_head_self_attention_8/dense_74/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=transformer_block_8/multi_head_self_attention_8/dense_74/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?transformer_block_8/multi_head_self_attention_8/dense_75/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=transformer_block_8/multi_head_self_attention_8/dense_75/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_76/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_76/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_77/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_77/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_8/layer_normalization_16/gamma1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_8/layer_normalization_16/beta1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_8/layer_normalization_17/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_8/layer_normalization_17/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
 
(
?0
?1
?2
?3
?4
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
 

M0

M0
?
dregularization_losses
?non_trainable_variables
etrainable_variables
?layer_metrics
f	variables
?metrics
?layers
 ?layer_regularization_losses
 

N0

N0
?
hregularization_losses
?non_trainable_variables
itrainable_variables
?layer_metrics
j	variables
?metrics
?layers
 ?layer_regularization_losses
 
 
 

0
1
 
l

Okernel
Pbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Qkernel
Rbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Skernel
Tbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Ukernel
Vbias
?regularization_losses
?trainable_variables
?	variables
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
uregularization_losses
?non_trainable_variables
vtrainable_variables
?layer_metrics
w	variables
?metrics
?layers
 ?layer_regularization_losses
l

Wkernel
Xbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

Ykernel
Zbias
?regularization_losses
?trainable_variables
?	variables
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
{regularization_losses
?non_trainable_variables
|trainable_variables
?layer_metrics
}	variables
?metrics
?layers
 ?layer_regularization_losses
 
 

[0
\1

[0
\1
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
 
 

]0
^1

]0
^1
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
 
 
 
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
 
 
 
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
 

Q0
R1

Q0
R1
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
 

S0
T1

S0
T1
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
 

U0
V1

U0
V1
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
 
 
 

q0
r1
s2
t3
 
 

W0
X1

W0
X1
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
 

Y0
Z1

Y0
Z1
?
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
VARIABLE_VALUEAdam/dense_78/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_78/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_79/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_79/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_80/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_80/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/main_output/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/main_output/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=Adam/token_and_position_embedding_8/embedding_16/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=Adam/token_and_position_embedding_8/embedding_17/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_76/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_76/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_77/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_77/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_16/gamma/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_16/beta/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_17/gamma/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_17/beta/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/aux_output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/aux_output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_78/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_78/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_79/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_79/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_80/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_80/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/main_output/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/main_output/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=Adam/token_and_position_embedding_8/embedding_16/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=Adam/token_and_position_embedding_8/embedding_17/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEFAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEDAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_76/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_76/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_77/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_77/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_16/gamma/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_16/beta/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_17/gamma/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_17/beta/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_aux_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_9Placeholder*'
_output_shapes
:?????????(*
dtype0*
shape:?????????(
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_aux_inputserving_default_input_96token_and_position_embedding_8/embedding_17/embeddings6token_and_position_embedding_8/embedding_16/embeddings?transformer_block_8/multi_head_self_attention_8/dense_72/kernel=transformer_block_8/multi_head_self_attention_8/dense_72/bias?transformer_block_8/multi_head_self_attention_8/dense_73/kernel=transformer_block_8/multi_head_self_attention_8/dense_73/bias?transformer_block_8/multi_head_self_attention_8/dense_74/kernel=transformer_block_8/multi_head_self_attention_8/dense_74/bias?transformer_block_8/multi_head_self_attention_8/dense_75/kernel=transformer_block_8/multi_head_self_attention_8/dense_75/bias0transformer_block_8/layer_normalization_16/gamma/transformer_block_8/layer_normalization_16/betadense_76/kerneldense_76/biasdense_77/kerneldense_77/bias0transformer_block_8/layer_normalization_17/gamma/transformer_block_8/layer_normalization_17/betaaux_output/kernelaux_output/biasdense_78/kerneldense_78/biasdense_79/kerneldense_79/biasdense_80/kerneldense_80/biasmain_output/kernelmain_output/bias*)
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
%__inference_signature_wrapper_4833103
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?/
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%aux_output/kernel/Read/ReadVariableOp#aux_output/bias/Read/ReadVariableOp#dense_78/kernel/Read/ReadVariableOp!dense_78/bias/Read/ReadVariableOp#dense_79/kernel/Read/ReadVariableOp!dense_79/bias/Read/ReadVariableOp#dense_80/kernel/Read/ReadVariableOp!dense_80/bias/Read/ReadVariableOp&main_output/kernel/Read/ReadVariableOp$main_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpJtoken_and_position_embedding_8/embedding_16/embeddings/Read/ReadVariableOpJtoken_and_position_embedding_8/embedding_17/embeddings/Read/ReadVariableOpStransformer_block_8/multi_head_self_attention_8/dense_72/kernel/Read/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_72/bias/Read/ReadVariableOpStransformer_block_8/multi_head_self_attention_8/dense_73/kernel/Read/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_73/bias/Read/ReadVariableOpStransformer_block_8/multi_head_self_attention_8/dense_74/kernel/Read/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_74/bias/Read/ReadVariableOpStransformer_block_8/multi_head_self_attention_8/dense_75/kernel/Read/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_75/bias/Read/ReadVariableOp#dense_76/kernel/Read/ReadVariableOp!dense_76/bias/Read/ReadVariableOp#dense_77/kernel/Read/ReadVariableOp!dense_77/bias/Read/ReadVariableOpDtransformer_block_8/layer_normalization_16/gamma/Read/ReadVariableOpCtransformer_block_8/layer_normalization_16/beta/Read/ReadVariableOpDtransformer_block_8/layer_normalization_17/gamma/Read/ReadVariableOpCtransformer_block_8/layer_normalization_17/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOp,Adam/aux_output/kernel/m/Read/ReadVariableOp*Adam/aux_output/bias/m/Read/ReadVariableOp*Adam/dense_78/kernel/m/Read/ReadVariableOp(Adam/dense_78/bias/m/Read/ReadVariableOp*Adam/dense_79/kernel/m/Read/ReadVariableOp(Adam/dense_79/bias/m/Read/ReadVariableOp*Adam/dense_80/kernel/m/Read/ReadVariableOp(Adam/dense_80/bias/m/Read/ReadVariableOp-Adam/main_output/kernel/m/Read/ReadVariableOp+Adam/main_output/bias/m/Read/ReadVariableOpQAdam/token_and_position_embedding_8/embedding_16/embeddings/m/Read/ReadVariableOpQAdam/token_and_position_embedding_8/embedding_17/embeddings/m/Read/ReadVariableOpZAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/m/Read/ReadVariableOpXAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/m/Read/ReadVariableOpZAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/m/Read/ReadVariableOpXAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/m/Read/ReadVariableOpZAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/m/Read/ReadVariableOpXAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/m/Read/ReadVariableOpZAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/m/Read/ReadVariableOpXAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/m/Read/ReadVariableOp*Adam/dense_76/kernel/m/Read/ReadVariableOp(Adam/dense_76/bias/m/Read/ReadVariableOp*Adam/dense_77/kernel/m/Read/ReadVariableOp(Adam/dense_77/bias/m/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_16/gamma/m/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_16/beta/m/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_17/gamma/m/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_17/beta/m/Read/ReadVariableOp,Adam/aux_output/kernel/v/Read/ReadVariableOp*Adam/aux_output/bias/v/Read/ReadVariableOp*Adam/dense_78/kernel/v/Read/ReadVariableOp(Adam/dense_78/bias/v/Read/ReadVariableOp*Adam/dense_79/kernel/v/Read/ReadVariableOp(Adam/dense_79/bias/v/Read/ReadVariableOp*Adam/dense_80/kernel/v/Read/ReadVariableOp(Adam/dense_80/bias/v/Read/ReadVariableOp-Adam/main_output/kernel/v/Read/ReadVariableOp+Adam/main_output/bias/v/Read/ReadVariableOpQAdam/token_and_position_embedding_8/embedding_16/embeddings/v/Read/ReadVariableOpQAdam/token_and_position_embedding_8/embedding_17/embeddings/v/Read/ReadVariableOpZAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/v/Read/ReadVariableOpXAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/v/Read/ReadVariableOpZAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/v/Read/ReadVariableOpXAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/v/Read/ReadVariableOpZAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/v/Read/ReadVariableOpXAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/v/Read/ReadVariableOpZAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/v/Read/ReadVariableOpXAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/v/Read/ReadVariableOp*Adam/dense_76/kernel/v/Read/ReadVariableOp(Adam/dense_76/bias/v/Read/ReadVariableOp*Adam/dense_77/kernel/v/Read/ReadVariableOp(Adam/dense_77/bias/v/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_16/gamma/v/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_16/beta/v/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_17/gamma/v/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_17/beta/v/Read/ReadVariableOpConst*p
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
 __inference__traced_save_4835140
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameaux_output/kernelaux_output/biasdense_78/kerneldense_78/biasdense_79/kerneldense_79/biasdense_80/kerneldense_80/biasmain_output/kernelmain_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate6token_and_position_embedding_8/embedding_16/embeddings6token_and_position_embedding_8/embedding_17/embeddings?transformer_block_8/multi_head_self_attention_8/dense_72/kernel=transformer_block_8/multi_head_self_attention_8/dense_72/bias?transformer_block_8/multi_head_self_attention_8/dense_73/kernel=transformer_block_8/multi_head_self_attention_8/dense_73/bias?transformer_block_8/multi_head_self_attention_8/dense_74/kernel=transformer_block_8/multi_head_self_attention_8/dense_74/bias?transformer_block_8/multi_head_self_attention_8/dense_75/kernel=transformer_block_8/multi_head_self_attention_8/dense_75/biasdense_76/kerneldense_76/biasdense_77/kerneldense_77/bias0transformer_block_8/layer_normalization_16/gamma/transformer_block_8/layer_normalization_16/beta0transformer_block_8/layer_normalization_17/gamma/transformer_block_8/layer_normalization_17/betatotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4Adam/aux_output/kernel/mAdam/aux_output/bias/mAdam/dense_78/kernel/mAdam/dense_78/bias/mAdam/dense_79/kernel/mAdam/dense_79/bias/mAdam/dense_80/kernel/mAdam/dense_80/bias/mAdam/main_output/kernel/mAdam/main_output/bias/m=Adam/token_and_position_embedding_8/embedding_16/embeddings/m=Adam/token_and_position_embedding_8/embedding_17/embeddings/mFAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/mDAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/mFAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/mDAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/mFAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/mDAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/mFAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/mDAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/mAdam/dense_76/kernel/mAdam/dense_76/bias/mAdam/dense_77/kernel/mAdam/dense_77/bias/m7Adam/transformer_block_8/layer_normalization_16/gamma/m6Adam/transformer_block_8/layer_normalization_16/beta/m7Adam/transformer_block_8/layer_normalization_17/gamma/m6Adam/transformer_block_8/layer_normalization_17/beta/mAdam/aux_output/kernel/vAdam/aux_output/bias/vAdam/dense_78/kernel/vAdam/dense_78/bias/vAdam/dense_79/kernel/vAdam/dense_79/bias/vAdam/dense_80/kernel/vAdam/dense_80/bias/vAdam/main_output/kernel/vAdam/main_output/bias/v=Adam/token_and_position_embedding_8/embedding_16/embeddings/v=Adam/token_and_position_embedding_8/embedding_17/embeddings/vFAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/vDAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/vFAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/vDAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/vFAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/vDAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/vFAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/vDAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/vAdam/dense_76/kernel/vAdam/dense_76/bias/vAdam/dense_77/kernel/vAdam/dense_77/bias/v7Adam/transformer_block_8/layer_normalization_16/gamma/v6Adam/transformer_block_8/layer_normalization_16/beta/v7Adam/transformer_block_8/layer_normalization_17/gamma/v6Adam/transformer_block_8/layer_normalization_17/beta/v*o
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
#__inference__traced_restore_4835447??,
?
?
E__inference_dense_78_layer_call_and_return_conditional_losses_4832101

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
?
?
H__inference_main_output_layer_call_and_return_conditional_losses_4834599

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
E__inference_dense_80_layer_call_and_return_conditional_losses_4832135

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
?
t
J__inference_concatenate_8_layer_call_and_return_conditional_losses_4832088

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
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_aux_output_layer_call_and_return_conditional_losses_4834506

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
.__inference_sequential_8_layer_call_fn_4834612

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
I__inference_sequential_8_layer_call_and_return_conditional_losses_48316042
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
E__inference_dense_79_layer_call_and_return_conditional_losses_4832118

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
?
X
<__inference_global_average_pooling1d_8_layer_call_fn_4834469

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
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_48317262
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
?
s
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_4832062

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
?
?
E__inference_dense_78_layer_call_and_return_conditional_losses_4834539

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
?
?
I__inference_sequential_8_layer_call_and_return_conditional_losses_4831604

inputs"
dense_76_4831562:  
dense_76_4831564: "
dense_77_4831598:  
dense_77_4831600: 
identity?? dense_76/StatefulPartitionedCall? dense_77/StatefulPartitionedCall?
 dense_76/StatefulPartitionedCallStatefulPartitionedCallinputsdense_76_4831562dense_76_4831564*
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
E__inference_dense_76_layer_call_and_return_conditional_losses_48315612"
 dense_76/StatefulPartitionedCall?
 dense_77/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0dense_77_4831598dense_77_4831600*
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
E__inference_dense_77_layer_call_and_return_conditional_losses_48315972"
 dense_77/StatefulPartitionedCall?
IdentityIdentity)dense_77/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp!^dense_76/StatefulPartitionedCall!^dense_77/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2D
 dense_77/StatefulPartitionedCall dense_77/StatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?K
?
I__inference_sequential_8_layer_call_and_return_conditional_losses_4834739

inputs<
*dense_76_tensordot_readvariableop_resource:  6
(dense_76_biasadd_readvariableop_resource: <
*dense_77_tensordot_readvariableop_resource:  6
(dense_77_biasadd_readvariableop_resource: 
identity??dense_76/BiasAdd/ReadVariableOp?!dense_76/Tensordot/ReadVariableOp?dense_77/BiasAdd/ReadVariableOp?!dense_77/Tensordot/ReadVariableOp?
!dense_76/Tensordot/ReadVariableOpReadVariableOp*dense_76_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_76/Tensordot/ReadVariableOp|
dense_76/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_76/Tensordot/axes?
dense_76/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_76/Tensordot/freej
dense_76/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_76/Tensordot/Shape?
 dense_76/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_76/Tensordot/GatherV2/axis?
dense_76/Tensordot/GatherV2GatherV2!dense_76/Tensordot/Shape:output:0 dense_76/Tensordot/free:output:0)dense_76/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_76/Tensordot/GatherV2?
"dense_76/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_76/Tensordot/GatherV2_1/axis?
dense_76/Tensordot/GatherV2_1GatherV2!dense_76/Tensordot/Shape:output:0 dense_76/Tensordot/axes:output:0+dense_76/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_76/Tensordot/GatherV2_1~
dense_76/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_76/Tensordot/Const?
dense_76/Tensordot/ProdProd$dense_76/Tensordot/GatherV2:output:0!dense_76/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_76/Tensordot/Prod?
dense_76/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_76/Tensordot/Const_1?
dense_76/Tensordot/Prod_1Prod&dense_76/Tensordot/GatherV2_1:output:0#dense_76/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_76/Tensordot/Prod_1?
dense_76/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_76/Tensordot/concat/axis?
dense_76/Tensordot/concatConcatV2 dense_76/Tensordot/free:output:0 dense_76/Tensordot/axes:output:0'dense_76/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_76/Tensordot/concat?
dense_76/Tensordot/stackPack dense_76/Tensordot/Prod:output:0"dense_76/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_76/Tensordot/stack?
dense_76/Tensordot/transpose	Transposeinputs"dense_76/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_76/Tensordot/transpose?
dense_76/Tensordot/ReshapeReshape dense_76/Tensordot/transpose:y:0!dense_76/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_76/Tensordot/Reshape?
dense_76/Tensordot/MatMulMatMul#dense_76/Tensordot/Reshape:output:0)dense_76/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_76/Tensordot/MatMul?
dense_76/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_76/Tensordot/Const_2?
 dense_76/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_76/Tensordot/concat_1/axis?
dense_76/Tensordot/concat_1ConcatV2$dense_76/Tensordot/GatherV2:output:0#dense_76/Tensordot/Const_2:output:0)dense_76/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_76/Tensordot/concat_1?
dense_76/TensordotReshape#dense_76/Tensordot/MatMul:product:0$dense_76/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_76/Tensordot?
dense_76/BiasAdd/ReadVariableOpReadVariableOp(dense_76_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_76/BiasAdd/ReadVariableOp?
dense_76/BiasAddBiasAdddense_76/Tensordot:output:0'dense_76/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_76/BiasAddw
dense_76/ReluReludense_76/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dense_76/Relu?
!dense_77/Tensordot/ReadVariableOpReadVariableOp*dense_77_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_77/Tensordot/ReadVariableOp|
dense_77/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_77/Tensordot/axes?
dense_77/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_77/Tensordot/free
dense_77/Tensordot/ShapeShapedense_76/Relu:activations:0*
T0*
_output_shapes
:2
dense_77/Tensordot/Shape?
 dense_77/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_77/Tensordot/GatherV2/axis?
dense_77/Tensordot/GatherV2GatherV2!dense_77/Tensordot/Shape:output:0 dense_77/Tensordot/free:output:0)dense_77/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_77/Tensordot/GatherV2?
"dense_77/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_77/Tensordot/GatherV2_1/axis?
dense_77/Tensordot/GatherV2_1GatherV2!dense_77/Tensordot/Shape:output:0 dense_77/Tensordot/axes:output:0+dense_77/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_77/Tensordot/GatherV2_1~
dense_77/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_77/Tensordot/Const?
dense_77/Tensordot/ProdProd$dense_77/Tensordot/GatherV2:output:0!dense_77/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_77/Tensordot/Prod?
dense_77/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_77/Tensordot/Const_1?
dense_77/Tensordot/Prod_1Prod&dense_77/Tensordot/GatherV2_1:output:0#dense_77/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_77/Tensordot/Prod_1?
dense_77/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_77/Tensordot/concat/axis?
dense_77/Tensordot/concatConcatV2 dense_77/Tensordot/free:output:0 dense_77/Tensordot/axes:output:0'dense_77/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_77/Tensordot/concat?
dense_77/Tensordot/stackPack dense_77/Tensordot/Prod:output:0"dense_77/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_77/Tensordot/stack?
dense_77/Tensordot/transpose	Transposedense_76/Relu:activations:0"dense_77/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_77/Tensordot/transpose?
dense_77/Tensordot/ReshapeReshape dense_77/Tensordot/transpose:y:0!dense_77/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_77/Tensordot/Reshape?
dense_77/Tensordot/MatMulMatMul#dense_77/Tensordot/Reshape:output:0)dense_77/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_77/Tensordot/MatMul?
dense_77/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_77/Tensordot/Const_2?
 dense_77/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_77/Tensordot/concat_1/axis?
dense_77/Tensordot/concat_1ConcatV2$dense_77/Tensordot/GatherV2:output:0#dense_77/Tensordot/Const_2:output:0)dense_77/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_77/Tensordot/concat_1?
dense_77/TensordotReshape#dense_77/Tensordot/MatMul:product:0$dense_77/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_77/Tensordot?
dense_77/BiasAdd/ReadVariableOpReadVariableOp(dense_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_77/BiasAdd/ReadVariableOp?
dense_77/BiasAddBiasAdddense_77/Tensordot:output:0'dense_77/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_77/BiasAddx
IdentityIdentitydense_77/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp ^dense_76/BiasAdd/ReadVariableOp"^dense_76/Tensordot/ReadVariableOp ^dense_77/BiasAdd/ReadVariableOp"^dense_77/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2B
dense_76/BiasAdd/ReadVariableOpdense_76/BiasAdd/ReadVariableOp2F
!dense_76/Tensordot/ReadVariableOp!dense_76/Tensordot/ReadVariableOp2B
dense_77/BiasAdd/ReadVariableOpdense_77/BiasAdd/ReadVariableOp2F
!dense_77/Tensordot/ReadVariableOp!dense_77/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?K
?
I__inference_sequential_8_layer_call_and_return_conditional_losses_4834682

inputs<
*dense_76_tensordot_readvariableop_resource:  6
(dense_76_biasadd_readvariableop_resource: <
*dense_77_tensordot_readvariableop_resource:  6
(dense_77_biasadd_readvariableop_resource: 
identity??dense_76/BiasAdd/ReadVariableOp?!dense_76/Tensordot/ReadVariableOp?dense_77/BiasAdd/ReadVariableOp?!dense_77/Tensordot/ReadVariableOp?
!dense_76/Tensordot/ReadVariableOpReadVariableOp*dense_76_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_76/Tensordot/ReadVariableOp|
dense_76/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_76/Tensordot/axes?
dense_76/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_76/Tensordot/freej
dense_76/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_76/Tensordot/Shape?
 dense_76/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_76/Tensordot/GatherV2/axis?
dense_76/Tensordot/GatherV2GatherV2!dense_76/Tensordot/Shape:output:0 dense_76/Tensordot/free:output:0)dense_76/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_76/Tensordot/GatherV2?
"dense_76/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_76/Tensordot/GatherV2_1/axis?
dense_76/Tensordot/GatherV2_1GatherV2!dense_76/Tensordot/Shape:output:0 dense_76/Tensordot/axes:output:0+dense_76/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_76/Tensordot/GatherV2_1~
dense_76/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_76/Tensordot/Const?
dense_76/Tensordot/ProdProd$dense_76/Tensordot/GatherV2:output:0!dense_76/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_76/Tensordot/Prod?
dense_76/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_76/Tensordot/Const_1?
dense_76/Tensordot/Prod_1Prod&dense_76/Tensordot/GatherV2_1:output:0#dense_76/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_76/Tensordot/Prod_1?
dense_76/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_76/Tensordot/concat/axis?
dense_76/Tensordot/concatConcatV2 dense_76/Tensordot/free:output:0 dense_76/Tensordot/axes:output:0'dense_76/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_76/Tensordot/concat?
dense_76/Tensordot/stackPack dense_76/Tensordot/Prod:output:0"dense_76/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_76/Tensordot/stack?
dense_76/Tensordot/transpose	Transposeinputs"dense_76/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_76/Tensordot/transpose?
dense_76/Tensordot/ReshapeReshape dense_76/Tensordot/transpose:y:0!dense_76/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_76/Tensordot/Reshape?
dense_76/Tensordot/MatMulMatMul#dense_76/Tensordot/Reshape:output:0)dense_76/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_76/Tensordot/MatMul?
dense_76/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_76/Tensordot/Const_2?
 dense_76/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_76/Tensordot/concat_1/axis?
dense_76/Tensordot/concat_1ConcatV2$dense_76/Tensordot/GatherV2:output:0#dense_76/Tensordot/Const_2:output:0)dense_76/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_76/Tensordot/concat_1?
dense_76/TensordotReshape#dense_76/Tensordot/MatMul:product:0$dense_76/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_76/Tensordot?
dense_76/BiasAdd/ReadVariableOpReadVariableOp(dense_76_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_76/BiasAdd/ReadVariableOp?
dense_76/BiasAddBiasAdddense_76/Tensordot:output:0'dense_76/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_76/BiasAddw
dense_76/ReluReludense_76/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dense_76/Relu?
!dense_77/Tensordot/ReadVariableOpReadVariableOp*dense_77_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_77/Tensordot/ReadVariableOp|
dense_77/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_77/Tensordot/axes?
dense_77/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_77/Tensordot/free
dense_77/Tensordot/ShapeShapedense_76/Relu:activations:0*
T0*
_output_shapes
:2
dense_77/Tensordot/Shape?
 dense_77/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_77/Tensordot/GatherV2/axis?
dense_77/Tensordot/GatherV2GatherV2!dense_77/Tensordot/Shape:output:0 dense_77/Tensordot/free:output:0)dense_77/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_77/Tensordot/GatherV2?
"dense_77/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_77/Tensordot/GatherV2_1/axis?
dense_77/Tensordot/GatherV2_1GatherV2!dense_77/Tensordot/Shape:output:0 dense_77/Tensordot/axes:output:0+dense_77/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_77/Tensordot/GatherV2_1~
dense_77/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_77/Tensordot/Const?
dense_77/Tensordot/ProdProd$dense_77/Tensordot/GatherV2:output:0!dense_77/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_77/Tensordot/Prod?
dense_77/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_77/Tensordot/Const_1?
dense_77/Tensordot/Prod_1Prod&dense_77/Tensordot/GatherV2_1:output:0#dense_77/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_77/Tensordot/Prod_1?
dense_77/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_77/Tensordot/concat/axis?
dense_77/Tensordot/concatConcatV2 dense_77/Tensordot/free:output:0 dense_77/Tensordot/axes:output:0'dense_77/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_77/Tensordot/concat?
dense_77/Tensordot/stackPack dense_77/Tensordot/Prod:output:0"dense_77/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_77/Tensordot/stack?
dense_77/Tensordot/transpose	Transposedense_76/Relu:activations:0"dense_77/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2
dense_77/Tensordot/transpose?
dense_77/Tensordot/ReshapeReshape dense_77/Tensordot/transpose:y:0!dense_77/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_77/Tensordot/Reshape?
dense_77/Tensordot/MatMulMatMul#dense_77/Tensordot/Reshape:output:0)dense_77/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_77/Tensordot/MatMul?
dense_77/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_77/Tensordot/Const_2?
 dense_77/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_77/Tensordot/concat_1/axis?
dense_77/Tensordot/concat_1ConcatV2$dense_77/Tensordot/GatherV2:output:0#dense_77/Tensordot/Const_2:output:0)dense_77/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_77/Tensordot/concat_1?
dense_77/TensordotReshape#dense_77/Tensordot/MatMul:product:0$dense_77/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2
dense_77/Tensordot?
dense_77/BiasAdd/ReadVariableOpReadVariableOp(dense_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_77/BiasAdd/ReadVariableOp?
dense_77/BiasAddBiasAdddense_77/Tensordot:output:0'dense_77/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
dense_77/BiasAddx
IdentityIdentitydense_77/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp ^dense_76/BiasAdd/ReadVariableOp"^dense_76/Tensordot/ReadVariableOp ^dense_77/BiasAdd/ReadVariableOp"^dense_77/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2B
dense_76/BiasAdd/ReadVariableOpdense_76/BiasAdd/ReadVariableOp2F
!dense_76/Tensordot/ReadVariableOp!dense_76/Tensordot/ReadVariableOp2B
dense_77/BiasAdd/ReadVariableOpdense_77/BiasAdd/ReadVariableOp2F
!dense_77/Tensordot/ReadVariableOp!dense_77/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_4833888
x7
%embedding_17_embedding_lookup_4833875:( 7
%embedding_16_embedding_lookup_4833881: 
identity??embedding_16/embedding_lookup?embedding_17/embedding_lookup?
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
embedding_17/embedding_lookupResourceGather%embedding_17_embedding_lookup_4833875range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_17/embedding_lookup/4833875*'
_output_shapes
:????????? *
dtype02
embedding_17/embedding_lookup?
&embedding_17/embedding_lookup/IdentityIdentity&embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_17/embedding_lookup/4833875*'
_output_shapes
:????????? 2(
&embedding_17/embedding_lookup/Identity?
(embedding_17/embedding_lookup/Identity_1Identity/embedding_17/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2*
(embedding_17/embedding_lookup/Identity_1r
embedding_16/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:?????????(2
embedding_16/Cast?
embedding_16/embedding_lookupResourceGather%embedding_16_embedding_lookup_4833881embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_16/embedding_lookup/4833881*+
_output_shapes
:?????????( *
dtype02
embedding_16/embedding_lookup?
&embedding_16/embedding_lookup/IdentityIdentity&embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_16/embedding_lookup/4833881*+
_output_shapes
:?????????( 2(
&embedding_16/embedding_lookup/Identity?
(embedding_16/embedding_lookup/Identity_1Identity/embedding_16/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2*
(embedding_16/embedding_lookup/Identity_1?
addAddV21embedding_16/embedding_lookup/Identity_1:output:01embedding_17/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2
addf
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp^embedding_16/embedding_lookup^embedding_17/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 2>
embedding_16/embedding_lookupembedding_16/embedding_lookup2>
embedding_17/embedding_lookupembedding_17/embedding_lookup:J F
'
_output_shapes
:?????????(

_user_specified_namex
?
?
I__inference_sequential_8_layer_call_and_return_conditional_losses_4831716
dense_76_input"
dense_76_4831705:  
dense_76_4831707: "
dense_77_4831710:  
dense_77_4831712: 
identity?? dense_76/StatefulPartitionedCall? dense_77/StatefulPartitionedCall?
 dense_76/StatefulPartitionedCallStatefulPartitionedCalldense_76_inputdense_76_4831705dense_76_4831707*
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
E__inference_dense_76_layer_call_and_return_conditional_losses_48315612"
 dense_76/StatefulPartitionedCall?
 dense_77/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0dense_77_4831710dense_77_4831712*
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
E__inference_dense_77_layer_call_and_return_conditional_losses_48315972"
 dense_77/StatefulPartitionedCall?
IdentityIdentity)dense_77/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp!^dense_76/StatefulPartitionedCall!^dense_77/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2D
 dense_77/StatefulPartitionedCall dense_77/StatefulPartitionedCall:[ W
+
_output_shapes
:?????????( 
(
_user_specified_namedense_76_input
??
? 
D__inference_model_8_layer_call_and_return_conditional_losses_4833855
inputs_0
inputs_1V
Dtoken_and_position_embedding_8_embedding_17_embedding_lookup_4833548:( V
Dtoken_and_position_embedding_8_embedding_16_embedding_lookup_4833554: l
Ztransformer_block_8_multi_head_self_attention_8_dense_72_tensordot_readvariableop_resource:  f
Xtransformer_block_8_multi_head_self_attention_8_dense_72_biasadd_readvariableop_resource: l
Ztransformer_block_8_multi_head_self_attention_8_dense_73_tensordot_readvariableop_resource:  f
Xtransformer_block_8_multi_head_self_attention_8_dense_73_biasadd_readvariableop_resource: l
Ztransformer_block_8_multi_head_self_attention_8_dense_74_tensordot_readvariableop_resource:  f
Xtransformer_block_8_multi_head_self_attention_8_dense_74_biasadd_readvariableop_resource: l
Ztransformer_block_8_multi_head_self_attention_8_dense_75_tensordot_readvariableop_resource:  f
Xtransformer_block_8_multi_head_self_attention_8_dense_75_biasadd_readvariableop_resource: ^
Ptransformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource: Z
Ltransformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource: ]
Ktransformer_block_8_sequential_8_dense_76_tensordot_readvariableop_resource:  W
Itransformer_block_8_sequential_8_dense_76_biasadd_readvariableop_resource: ]
Ktransformer_block_8_sequential_8_dense_77_tensordot_readvariableop_resource:  W
Itransformer_block_8_sequential_8_dense_77_biasadd_readvariableop_resource: ^
Ptransformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource: Z
Ltransformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource: ;
)aux_output_matmul_readvariableop_resource: 8
*aux_output_biasadd_readvariableop_resource:9
'dense_78_matmul_readvariableop_resource:@6
(dense_78_biasadd_readvariableop_resource:@9
'dense_79_matmul_readvariableop_resource:@@6
(dense_79_biasadd_readvariableop_resource:@9
'dense_80_matmul_readvariableop_resource:@@6
(dense_80_biasadd_readvariableop_resource:@<
*main_output_matmul_readvariableop_resource:@9
+main_output_biasadd_readvariableop_resource:
identity

identity_1??!aux_output/BiasAdd/ReadVariableOp? aux_output/MatMul/ReadVariableOp?dense_78/BiasAdd/ReadVariableOp?dense_78/MatMul/ReadVariableOp?dense_79/BiasAdd/ReadVariableOp?dense_79/MatMul/ReadVariableOp?dense_80/BiasAdd/ReadVariableOp?dense_80/MatMul/ReadVariableOp?"main_output/BiasAdd/ReadVariableOp?!main_output/MatMul/ReadVariableOp?<token_and_position_embedding_8/embedding_16/embedding_lookup?<token_and_position_embedding_8/embedding_17/embedding_lookup?Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp?Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp?Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp?Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp?Otransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?Otransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?Otransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?Otransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?@transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp?Btransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp?@transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp?Btransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp?
$token_and_position_embedding_8/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_8/Shape?
2token_and_position_embedding_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2token_and_position_embedding_8/strided_slice/stack?
4token_and_position_embedding_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_8/strided_slice/stack_1?
4token_and_position_embedding_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_8/strided_slice/stack_2?
,token_and_position_embedding_8/strided_sliceStridedSlice-token_and_position_embedding_8/Shape:output:0;token_and_position_embedding_8/strided_slice/stack:output:0=token_and_position_embedding_8/strided_slice/stack_1:output:0=token_and_position_embedding_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_8/strided_slice?
*token_and_position_embedding_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_8/range/start?
*token_and_position_embedding_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_8/range/delta?
$token_and_position_embedding_8/rangeRange3token_and_position_embedding_8/range/start:output:05token_and_position_embedding_8/strided_slice:output:03token_and_position_embedding_8/range/delta:output:0*#
_output_shapes
:?????????2&
$token_and_position_embedding_8/range?
<token_and_position_embedding_8/embedding_17/embedding_lookupResourceGatherDtoken_and_position_embedding_8_embedding_17_embedding_lookup_4833548-token_and_position_embedding_8/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_17/embedding_lookup/4833548*'
_output_shapes
:????????? *
dtype02>
<token_and_position_embedding_8/embedding_17/embedding_lookup?
Etoken_and_position_embedding_8/embedding_17/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_8/embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_17/embedding_lookup/4833548*'
_output_shapes
:????????? 2G
Etoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity?
Gtoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2I
Gtoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1?
0token_and_position_embedding_8/embedding_16/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????(22
0token_and_position_embedding_8/embedding_16/Cast?
<token_and_position_embedding_8/embedding_16/embedding_lookupResourceGatherDtoken_and_position_embedding_8_embedding_16_embedding_lookup_48335544token_and_position_embedding_8/embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_16/embedding_lookup/4833554*+
_output_shapes
:?????????( *
dtype02>
<token_and_position_embedding_8/embedding_16/embedding_lookup?
Etoken_and_position_embedding_8/embedding_16/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_8/embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_16/embedding_lookup/4833554*+
_output_shapes
:?????????( 2G
Etoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity?
Gtoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2I
Gtoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1?
"token_and_position_embedding_8/addAddV2Ptoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1:output:0Ptoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2$
"token_and_position_embedding_8/add?
5transformer_block_8/multi_head_self_attention_8/ShapeShape&token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:27
5transformer_block_8/multi_head_self_attention_8/Shape?
Ctransformer_block_8/multi_head_self_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_8/multi_head_self_attention_8/strided_slice/stack?
Etransformer_block_8/multi_head_self_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Etransformer_block_8/multi_head_self_attention_8/strided_slice/stack_1?
Etransformer_block_8/multi_head_self_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Etransformer_block_8/multi_head_self_attention_8/strided_slice/stack_2?
=transformer_block_8/multi_head_self_attention_8/strided_sliceStridedSlice>transformer_block_8/multi_head_self_attention_8/Shape:output:0Ltransformer_block_8/multi_head_self_attention_8/strided_slice/stack:output:0Ntransformer_block_8/multi_head_self_attention_8/strided_slice/stack_1:output:0Ntransformer_block_8/multi_head_self_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=transformer_block_8/multi_head_self_attention_8/strided_slice?
Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_8_multi_head_self_attention_8_dense_72_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axes?
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/free?
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ShapeShape&token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Shape?
Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/free:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2?
Rtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis?
Mtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axes:output:0[transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1?
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const?
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ProdProdTtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod?
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_1?
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod_1ProdVtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1:output:0Stransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod_1?
Ntransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat/axis?
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concatConcatV2Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/free:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axes:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat?
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/stackPackPtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod:output:0Rtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/stack?
Ltransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/transpose	Transpose&token_and_position_embedding_8/add:z:0Rtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/transpose?
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReshapeReshapePtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/transpose:y:0Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Reshape?
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/MatMulMatMulStransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Reshape:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/MatMul?
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_2?
Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1ConcatV2Ttransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0Stransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_2:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1?
Btransformer_block_8/multi_head_self_attention_8/dense_72/TensordotReshapeStransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/MatMul:product:0Ttransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot?
Otransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_8_multi_head_self_attention_8_dense_72_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?
@transformer_block_8/multi_head_self_attention_8/dense_72/BiasAddBiasAddKtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd?
Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_8_multi_head_self_attention_8_dense_73_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axes?
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/free?
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ShapeShape&token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Shape?
Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/free:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2?
Rtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis?
Mtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axes:output:0[transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1?
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const?
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ProdProdTtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod?
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_1?
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod_1ProdVtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1:output:0Stransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod_1?
Ntransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat/axis?
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concatConcatV2Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/free:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axes:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat?
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/stackPackPtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod:output:0Rtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/stack?
Ltransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/transpose	Transpose&token_and_position_embedding_8/add:z:0Rtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/transpose?
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReshapeReshapePtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/transpose:y:0Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Reshape?
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/MatMulMatMulStransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Reshape:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/MatMul?
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_2?
Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1ConcatV2Ttransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0Stransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_2:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1?
Btransformer_block_8/multi_head_self_attention_8/dense_73/TensordotReshapeStransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/MatMul:product:0Ttransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot?
Otransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_8_multi_head_self_attention_8_dense_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?
@transformer_block_8/multi_head_self_attention_8/dense_73/BiasAddBiasAddKtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd?
Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_8_multi_head_self_attention_8_dense_74_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axes?
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/free?
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ShapeShape&token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Shape?
Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/free:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2?
Rtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis?
Mtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axes:output:0[transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1?
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const?
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ProdProdTtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod?
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_1?
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod_1ProdVtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1:output:0Stransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod_1?
Ntransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat/axis?
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concatConcatV2Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/free:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axes:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat?
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/stackPackPtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod:output:0Rtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/stack?
Ltransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/transpose	Transpose&token_and_position_embedding_8/add:z:0Rtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/transpose?
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReshapeReshapePtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/transpose:y:0Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Reshape?
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/MatMulMatMulStransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Reshape:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/MatMul?
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_2?
Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1ConcatV2Ttransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0Stransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_2:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1?
Btransformer_block_8/multi_head_self_attention_8/dense_74/TensordotReshapeStransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/MatMul:product:0Ttransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot?
Otransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_8_multi_head_self_attention_8_dense_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?
@transformer_block_8/multi_head_self_attention_8/dense_74/BiasAddBiasAddKtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd?
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2A
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/1?
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2A
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/2?
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2A
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/3?
=transformer_block_8/multi_head_self_attention_8/Reshape/shapePackFtransformer_block_8/multi_head_self_attention_8/strided_slice:output:0Htransformer_block_8/multi_head_self_attention_8/Reshape/shape/1:output:0Htransformer_block_8/multi_head_self_attention_8/Reshape/shape/2:output:0Htransformer_block_8/multi_head_self_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_8/multi_head_self_attention_8/Reshape/shape?
7transformer_block_8/multi_head_self_attention_8/ReshapeReshapeItransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd:output:0Ftransformer_block_8/multi_head_self_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????29
7transformer_block_8/multi_head_self_attention_8/Reshape?
>transformer_block_8/multi_head_self_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2@
>transformer_block_8/multi_head_self_attention_8/transpose/perm?
9transformer_block_8/multi_head_self_attention_8/transpose	Transpose@transformer_block_8/multi_head_self_attention_8/Reshape:output:0Gtransformer_block_8/multi_head_self_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_8/multi_head_self_attention_8/transpose?
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/1?
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/2?
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/3?
?transformer_block_8/multi_head_self_attention_8/Reshape_1/shapePackFtransformer_block_8/multi_head_self_attention_8/strided_slice:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/1:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/2:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_8/multi_head_self_attention_8/Reshape_1/shape?
9transformer_block_8/multi_head_self_attention_8/Reshape_1ReshapeItransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd:output:0Htransformer_block_8/multi_head_self_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_8/multi_head_self_attention_8/Reshape_1?
@transformer_block_8/multi_head_self_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_8/multi_head_self_attention_8/transpose_1/perm?
;transformer_block_8/multi_head_self_attention_8/transpose_1	TransposeBtransformer_block_8/multi_head_self_attention_8/Reshape_1:output:0Itransformer_block_8/multi_head_self_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_8/multi_head_self_attention_8/transpose_1?
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/1?
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/2?
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/3?
?transformer_block_8/multi_head_self_attention_8/Reshape_2/shapePackFtransformer_block_8/multi_head_self_attention_8/strided_slice:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/1:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/2:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_8/multi_head_self_attention_8/Reshape_2/shape?
9transformer_block_8/multi_head_self_attention_8/Reshape_2ReshapeItransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd:output:0Htransformer_block_8/multi_head_self_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_8/multi_head_self_attention_8/Reshape_2?
@transformer_block_8/multi_head_self_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_8/multi_head_self_attention_8/transpose_2/perm?
;transformer_block_8/multi_head_self_attention_8/transpose_2	TransposeBtransformer_block_8/multi_head_self_attention_8/Reshape_2:output:0Itransformer_block_8/multi_head_self_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_8/multi_head_self_attention_8/transpose_2?
6transformer_block_8/multi_head_self_attention_8/MatMulBatchMatMulV2=transformer_block_8/multi_head_self_attention_8/transpose:y:0?transformer_block_8/multi_head_self_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(28
6transformer_block_8/multi_head_self_attention_8/MatMul?
7transformer_block_8/multi_head_self_attention_8/Shape_1Shape?transformer_block_8/multi_head_self_attention_8/transpose_1:y:0*
T0*
_output_shapes
:29
7transformer_block_8/multi_head_self_attention_8/Shape_1?
Etransformer_block_8/multi_head_self_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2G
Etransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack?
Gtransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gtransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_1?
Gtransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_2?
?transformer_block_8/multi_head_self_attention_8/strided_slice_1StridedSlice@transformer_block_8/multi_head_self_attention_8/Shape_1:output:0Ntransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack:output:0Ptransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_1:output:0Ptransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?transformer_block_8/multi_head_self_attention_8/strided_slice_1?
4transformer_block_8/multi_head_self_attention_8/CastCastHtransformer_block_8/multi_head_self_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 26
4transformer_block_8/multi_head_self_attention_8/Cast?
4transformer_block_8/multi_head_self_attention_8/SqrtSqrt8transformer_block_8/multi_head_self_attention_8/Cast:y:0*
T0*
_output_shapes
: 26
4transformer_block_8/multi_head_self_attention_8/Sqrt?
7transformer_block_8/multi_head_self_attention_8/truedivRealDiv?transformer_block_8/multi_head_self_attention_8/MatMul:output:08transformer_block_8/multi_head_self_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????29
7transformer_block_8/multi_head_self_attention_8/truediv?
7transformer_block_8/multi_head_self_attention_8/SoftmaxSoftmax;transformer_block_8/multi_head_self_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????29
7transformer_block_8/multi_head_self_attention_8/Softmax?
8transformer_block_8/multi_head_self_attention_8/MatMul_1BatchMatMulV2Atransformer_block_8/multi_head_self_attention_8/Softmax:softmax:0?transformer_block_8/multi_head_self_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2:
8transformer_block_8/multi_head_self_attention_8/MatMul_1?
@transformer_block_8/multi_head_self_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_8/multi_head_self_attention_8/transpose_3/perm?
;transformer_block_8/multi_head_self_attention_8/transpose_3	TransposeAtransformer_block_8/multi_head_self_attention_8/MatMul_1:output:0Itransformer_block_8/multi_head_self_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_8/multi_head_self_attention_8/transpose_3?
Atransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/1?
Atransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/2?
?transformer_block_8/multi_head_self_attention_8/Reshape_3/shapePackFtransformer_block_8/multi_head_self_attention_8/strided_slice:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/1:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_8/multi_head_self_attention_8/Reshape_3/shape?
9transformer_block_8/multi_head_self_attention_8/Reshape_3Reshape?transformer_block_8/multi_head_self_attention_8/transpose_3:y:0Htransformer_block_8/multi_head_self_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2;
9transformer_block_8/multi_head_self_attention_8/Reshape_3?
Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_8_multi_head_self_attention_8_dense_75_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axes?
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/free?
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ShapeShapeBtransformer_block_8/multi_head_self_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Shape?
Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/free:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2?
Rtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis?
Mtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axes:output:0[transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1?
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const?
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ProdProdTtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod?
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_1?
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod_1ProdVtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1:output:0Stransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod_1?
Ntransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat/axis?
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concatConcatV2Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/free:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axes:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat?
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/stackPackPtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod:output:0Rtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/stack?
Ltransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/transpose	TransposeBtransformer_block_8/multi_head_self_attention_8/Reshape_3:output:0Rtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2N
Ltransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/transpose?
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReshapeReshapePtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/transpose:y:0Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Reshape?
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/MatMulMatMulStransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Reshape:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/MatMul?
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_2?
Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1ConcatV2Ttransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0Stransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_2:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1?
Btransformer_block_8/multi_head_self_attention_8/dense_75/TensordotReshapeStransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/MatMul:product:0Ttransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 2D
Btransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot?
Otransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_8_multi_head_self_attention_8_dense_75_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?
@transformer_block_8/multi_head_self_attention_8/dense_75/BiasAddBiasAddKtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2B
@transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd?
,transformer_block_8/dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2.
,transformer_block_8/dropout_16/dropout/Const?
*transformer_block_8/dropout_16/dropout/MulMulItransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd:output:05transformer_block_8/dropout_16/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2,
*transformer_block_8/dropout_16/dropout/Mul?
,transformer_block_8/dropout_16/dropout/ShapeShapeItransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_8/dropout_16/dropout/Shape?
Ctransformer_block_8/dropout_16/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_8/dropout_16/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype02E
Ctransformer_block_8/dropout_16/dropout/random_uniform/RandomUniform?
5transformer_block_8/dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=27
5transformer_block_8/dropout_16/dropout/GreaterEqual/y?
3transformer_block_8/dropout_16/dropout/GreaterEqualGreaterEqualLtransformer_block_8/dropout_16/dropout/random_uniform/RandomUniform:output:0>transformer_block_8/dropout_16/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 25
3transformer_block_8/dropout_16/dropout/GreaterEqual?
+transformer_block_8/dropout_16/dropout/CastCast7transformer_block_8/dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2-
+transformer_block_8/dropout_16/dropout/Cast?
,transformer_block_8/dropout_16/dropout/Mul_1Mul.transformer_block_8/dropout_16/dropout/Mul:z:0/transformer_block_8/dropout_16/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2.
,transformer_block_8/dropout_16/dropout/Mul_1?
transformer_block_8/addAddV2&token_and_position_embedding_8/add:z:00transformer_block_8/dropout_16/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_8/add?
Itransformer_block_8/layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_8/layer_normalization_16/moments/mean/reduction_indices?
7transformer_block_8/layer_normalization_16/moments/meanMeantransformer_block_8/add:z:0Rtransformer_block_8/layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(29
7transformer_block_8/layer_normalization_16/moments/mean?
?transformer_block_8/layer_normalization_16/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2A
?transformer_block_8/layer_normalization_16/moments/StopGradient?
Dtransformer_block_8/layer_normalization_16/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add:z:0Htransformer_block_8/layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2F
Dtransformer_block_8/layer_normalization_16/moments/SquaredDifference?
Mtransformer_block_8/layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_8/layer_normalization_16/moments/variance/reduction_indices?
;transformer_block_8/layer_normalization_16/moments/varianceMeanHtransformer_block_8/layer_normalization_16/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2=
;transformer_block_8/layer_normalization_16/moments/variance?
:transformer_block_8/layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:transformer_block_8/layer_normalization_16/batchnorm/add/y?
8transformer_block_8/layer_normalization_16/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_16/moments/variance:output:0Ctransformer_block_8/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2:
8transformer_block_8/layer_normalization_16/batchnorm/add?
:transformer_block_8/layer_normalization_16/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2<
:transformer_block_8/layer_normalization_16/batchnorm/Rsqrt?
Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp?
8transformer_block_8/layer_normalization_16/batchnorm/mulMul>transformer_block_8/layer_normalization_16/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2:
8transformer_block_8/layer_normalization_16/batchnorm/mul?
:transformer_block_8/layer_normalization_16/batchnorm/mul_1Multransformer_block_8/add:z:0<transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_16/batchnorm/mul_1?
:transformer_block_8/layer_normalization_16/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_16/moments/mean:output:0<transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_16/batchnorm/mul_2?
Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp?
8transformer_block_8/layer_normalization_16/batchnorm/subSubKtransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2:
8transformer_block_8/layer_normalization_16/batchnorm/sub?
:transformer_block_8/layer_normalization_16/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_16/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_16/batchnorm/add_1?
Btransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_8_sequential_8_dense_76_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02D
Btransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp?
8transformer_block_8/sequential_8/dense_76/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_8/sequential_8/dense_76/Tensordot/axes?
8transformer_block_8/sequential_8/dense_76/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_8/sequential_8/dense_76/Tensordot/free?
9transformer_block_8/sequential_8/dense_76/Tensordot/ShapeShape>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_8/sequential_8/dense_76/Tensordot/Shape?
Atransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2/axis?
<transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2GatherV2Btransformer_block_8/sequential_8/dense_76/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_76/Tensordot/free:output:0Jtransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2?
Ctransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1/axis?
>transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1GatherV2Btransformer_block_8/sequential_8/dense_76/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_76/Tensordot/axes:output:0Ltransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1?
9transformer_block_8/sequential_8/dense_76/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_8/sequential_8/dense_76/Tensordot/Const?
8transformer_block_8/sequential_8/dense_76/Tensordot/ProdProdEtransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2:output:0Btransformer_block_8/sequential_8/dense_76/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_8/sequential_8/dense_76/Tensordot/Prod?
;transformer_block_8/sequential_8/dense_76/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_8/sequential_8/dense_76/Tensordot/Const_1?
:transformer_block_8/sequential_8/dense_76/Tensordot/Prod_1ProdGtransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1:output:0Dtransformer_block_8/sequential_8/dense_76/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_8/sequential_8/dense_76/Tensordot/Prod_1?
?transformer_block_8/sequential_8/dense_76/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_8/sequential_8/dense_76/Tensordot/concat/axis?
:transformer_block_8/sequential_8/dense_76/Tensordot/concatConcatV2Atransformer_block_8/sequential_8/dense_76/Tensordot/free:output:0Atransformer_block_8/sequential_8/dense_76/Tensordot/axes:output:0Htransformer_block_8/sequential_8/dense_76/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_76/Tensordot/concat?
9transformer_block_8/sequential_8/dense_76/Tensordot/stackPackAtransformer_block_8/sequential_8/dense_76/Tensordot/Prod:output:0Ctransformer_block_8/sequential_8/dense_76/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_8/sequential_8/dense_76/Tensordot/stack?
=transformer_block_8/sequential_8/dense_76/Tensordot/transpose	Transpose>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0Ctransformer_block_8/sequential_8/dense_76/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2?
=transformer_block_8/sequential_8/dense_76/Tensordot/transpose?
;transformer_block_8/sequential_8/dense_76/Tensordot/ReshapeReshapeAtransformer_block_8/sequential_8/dense_76/Tensordot/transpose:y:0Btransformer_block_8/sequential_8/dense_76/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_8/sequential_8/dense_76/Tensordot/Reshape?
:transformer_block_8/sequential_8/dense_76/Tensordot/MatMulMatMulDtransformer_block_8/sequential_8/dense_76/Tensordot/Reshape:output:0Jtransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_8/sequential_8/dense_76/Tensordot/MatMul?
;transformer_block_8/sequential_8/dense_76/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_8/sequential_8/dense_76/Tensordot/Const_2?
Atransformer_block_8/sequential_8/dense_76/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_8/sequential_8/dense_76/Tensordot/concat_1/axis?
<transformer_block_8/sequential_8/dense_76/Tensordot/concat_1ConcatV2Etransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2:output:0Dtransformer_block_8/sequential_8/dense_76/Tensordot/Const_2:output:0Jtransformer_block_8/sequential_8/dense_76/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_8/sequential_8/dense_76/Tensordot/concat_1?
3transformer_block_8/sequential_8/dense_76/TensordotReshapeDtransformer_block_8/sequential_8/dense_76/Tensordot/MatMul:product:0Etransformer_block_8/sequential_8/dense_76/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 25
3transformer_block_8/sequential_8/dense_76/Tensordot?
@transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_8_sequential_8_dense_76_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp?
1transformer_block_8/sequential_8/dense_76/BiasAddBiasAdd<transformer_block_8/sequential_8/dense_76/Tensordot:output:0Htransformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 23
1transformer_block_8/sequential_8/dense_76/BiasAdd?
.transformer_block_8/sequential_8/dense_76/ReluRelu:transformer_block_8/sequential_8/dense_76/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 20
.transformer_block_8/sequential_8/dense_76/Relu?
Btransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_8_sequential_8_dense_77_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02D
Btransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp?
8transformer_block_8/sequential_8/dense_77/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_8/sequential_8/dense_77/Tensordot/axes?
8transformer_block_8/sequential_8/dense_77/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_8/sequential_8/dense_77/Tensordot/free?
9transformer_block_8/sequential_8/dense_77/Tensordot/ShapeShape<transformer_block_8/sequential_8/dense_76/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_8/sequential_8/dense_77/Tensordot/Shape?
Atransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2/axis?
<transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2GatherV2Btransformer_block_8/sequential_8/dense_77/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_77/Tensordot/free:output:0Jtransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2?
Ctransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1/axis?
>transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1GatherV2Btransformer_block_8/sequential_8/dense_77/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_77/Tensordot/axes:output:0Ltransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1?
9transformer_block_8/sequential_8/dense_77/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_8/sequential_8/dense_77/Tensordot/Const?
8transformer_block_8/sequential_8/dense_77/Tensordot/ProdProdEtransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2:output:0Btransformer_block_8/sequential_8/dense_77/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_8/sequential_8/dense_77/Tensordot/Prod?
;transformer_block_8/sequential_8/dense_77/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_8/sequential_8/dense_77/Tensordot/Const_1?
:transformer_block_8/sequential_8/dense_77/Tensordot/Prod_1ProdGtransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1:output:0Dtransformer_block_8/sequential_8/dense_77/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_8/sequential_8/dense_77/Tensordot/Prod_1?
?transformer_block_8/sequential_8/dense_77/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_8/sequential_8/dense_77/Tensordot/concat/axis?
:transformer_block_8/sequential_8/dense_77/Tensordot/concatConcatV2Atransformer_block_8/sequential_8/dense_77/Tensordot/free:output:0Atransformer_block_8/sequential_8/dense_77/Tensordot/axes:output:0Htransformer_block_8/sequential_8/dense_77/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_77/Tensordot/concat?
9transformer_block_8/sequential_8/dense_77/Tensordot/stackPackAtransformer_block_8/sequential_8/dense_77/Tensordot/Prod:output:0Ctransformer_block_8/sequential_8/dense_77/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_8/sequential_8/dense_77/Tensordot/stack?
=transformer_block_8/sequential_8/dense_77/Tensordot/transpose	Transpose<transformer_block_8/sequential_8/dense_76/Relu:activations:0Ctransformer_block_8/sequential_8/dense_77/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2?
=transformer_block_8/sequential_8/dense_77/Tensordot/transpose?
;transformer_block_8/sequential_8/dense_77/Tensordot/ReshapeReshapeAtransformer_block_8/sequential_8/dense_77/Tensordot/transpose:y:0Btransformer_block_8/sequential_8/dense_77/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_8/sequential_8/dense_77/Tensordot/Reshape?
:transformer_block_8/sequential_8/dense_77/Tensordot/MatMulMatMulDtransformer_block_8/sequential_8/dense_77/Tensordot/Reshape:output:0Jtransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_8/sequential_8/dense_77/Tensordot/MatMul?
;transformer_block_8/sequential_8/dense_77/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_8/sequential_8/dense_77/Tensordot/Const_2?
Atransformer_block_8/sequential_8/dense_77/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_8/sequential_8/dense_77/Tensordot/concat_1/axis?
<transformer_block_8/sequential_8/dense_77/Tensordot/concat_1ConcatV2Etransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2:output:0Dtransformer_block_8/sequential_8/dense_77/Tensordot/Const_2:output:0Jtransformer_block_8/sequential_8/dense_77/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_8/sequential_8/dense_77/Tensordot/concat_1?
3transformer_block_8/sequential_8/dense_77/TensordotReshapeDtransformer_block_8/sequential_8/dense_77/Tensordot/MatMul:product:0Etransformer_block_8/sequential_8/dense_77/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 25
3transformer_block_8/sequential_8/dense_77/Tensordot?
@transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_8_sequential_8_dense_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp?
1transformer_block_8/sequential_8/dense_77/BiasAddBiasAdd<transformer_block_8/sequential_8/dense_77/Tensordot:output:0Htransformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 23
1transformer_block_8/sequential_8/dense_77/BiasAdd?
,transformer_block_8/dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2.
,transformer_block_8/dropout_17/dropout/Const?
*transformer_block_8/dropout_17/dropout/MulMul:transformer_block_8/sequential_8/dense_77/BiasAdd:output:05transformer_block_8/dropout_17/dropout/Const:output:0*
T0*+
_output_shapes
:?????????( 2,
*transformer_block_8/dropout_17/dropout/Mul?
,transformer_block_8/dropout_17/dropout/ShapeShape:transformer_block_8/sequential_8/dense_77/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_8/dropout_17/dropout/Shape?
Ctransformer_block_8/dropout_17/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_8/dropout_17/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????( *
dtype02E
Ctransformer_block_8/dropout_17/dropout/random_uniform/RandomUniform?
5transformer_block_8/dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=27
5transformer_block_8/dropout_17/dropout/GreaterEqual/y?
3transformer_block_8/dropout_17/dropout/GreaterEqualGreaterEqualLtransformer_block_8/dropout_17/dropout/random_uniform/RandomUniform:output:0>transformer_block_8/dropout_17/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????( 25
3transformer_block_8/dropout_17/dropout/GreaterEqual?
+transformer_block_8/dropout_17/dropout/CastCast7transformer_block_8/dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????( 2-
+transformer_block_8/dropout_17/dropout/Cast?
,transformer_block_8/dropout_17/dropout/Mul_1Mul.transformer_block_8/dropout_17/dropout/Mul:z:0/transformer_block_8/dropout_17/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????( 2.
,transformer_block_8/dropout_17/dropout/Mul_1?
transformer_block_8/add_1AddV2>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:00transformer_block_8/dropout_17/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_8/add_1?
Itransformer_block_8/layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_8/layer_normalization_17/moments/mean/reduction_indices?
7transformer_block_8/layer_normalization_17/moments/meanMeantransformer_block_8/add_1:z:0Rtransformer_block_8/layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(29
7transformer_block_8/layer_normalization_17/moments/mean?
?transformer_block_8/layer_normalization_17/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2A
?transformer_block_8/layer_normalization_17/moments/StopGradient?
Dtransformer_block_8/layer_normalization_17/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add_1:z:0Htransformer_block_8/layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2F
Dtransformer_block_8/layer_normalization_17/moments/SquaredDifference?
Mtransformer_block_8/layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_8/layer_normalization_17/moments/variance/reduction_indices?
;transformer_block_8/layer_normalization_17/moments/varianceMeanHtransformer_block_8/layer_normalization_17/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2=
;transformer_block_8/layer_normalization_17/moments/variance?
:transformer_block_8/layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:transformer_block_8/layer_normalization_17/batchnorm/add/y?
8transformer_block_8/layer_normalization_17/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_17/moments/variance:output:0Ctransformer_block_8/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2:
8transformer_block_8/layer_normalization_17/batchnorm/add?
:transformer_block_8/layer_normalization_17/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2<
:transformer_block_8/layer_normalization_17/batchnorm/Rsqrt?
Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp?
8transformer_block_8/layer_normalization_17/batchnorm/mulMul>transformer_block_8/layer_normalization_17/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2:
8transformer_block_8/layer_normalization_17/batchnorm/mul?
:transformer_block_8/layer_normalization_17/batchnorm/mul_1Multransformer_block_8/add_1:z:0<transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_17/batchnorm/mul_1?
:transformer_block_8/layer_normalization_17/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_17/moments/mean:output:0<transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_17/batchnorm/mul_2?
Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp?
8transformer_block_8/layer_normalization_17/batchnorm/subSubKtransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2:
8transformer_block_8/layer_normalization_17/batchnorm/sub?
:transformer_block_8/layer_normalization_17/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_17/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_17/batchnorm/add_1?
1global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_8/Mean/reduction_indices?
global_average_pooling1d_8/MeanMean>transformer_block_8/layer_normalization_17/batchnorm/add_1:z:0:global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2!
global_average_pooling1d_8/Mean?
 aux_output/MatMul/ReadVariableOpReadVariableOp)aux_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 aux_output/MatMul/ReadVariableOp?
aux_output/MatMulMatMul(global_average_pooling1d_8/Mean:output:0(aux_output/MatMul/ReadVariableOp:value:0*
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
concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_8/concat/axis?
concatenate_8/concatConcatV2aux_output/Sigmoid:y:0inputs_1"concatenate_8/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_8/concat?
dense_78/MatMul/ReadVariableOpReadVariableOp'dense_78_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_78/MatMul/ReadVariableOp?
dense_78/MatMulMatMulconcatenate_8/concat:output:0&dense_78/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_78/MatMul?
dense_78/BiasAdd/ReadVariableOpReadVariableOp(dense_78_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_78/BiasAdd/ReadVariableOp?
dense_78/BiasAddBiasAdddense_78/MatMul:product:0'dense_78/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_78/BiasAdds
dense_78/ReluReludense_78/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_78/Relu?
dense_79/MatMul/ReadVariableOpReadVariableOp'dense_79_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_79/MatMul/ReadVariableOp?
dense_79/MatMulMatMuldense_78/Relu:activations:0&dense_79/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_79/MatMul?
dense_79/BiasAdd/ReadVariableOpReadVariableOp(dense_79_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_79/BiasAdd/ReadVariableOp?
dense_79/BiasAddBiasAdddense_79/MatMul:product:0'dense_79/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_79/BiasAdds
dense_79/ReluReludense_79/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_79/Relu?
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_80/MatMul/ReadVariableOp?
dense_80/MatMulMatMuldense_79/Relu:activations:0&dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_80/MatMul?
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_80/BiasAdd/ReadVariableOp?
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_80/BiasAdds
dense_80/ReluReludense_80/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_80/Relu?
!main_output/MatMul/ReadVariableOpReadVariableOp*main_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!main_output/MatMul/ReadVariableOp?
main_output/MatMulMatMuldense_80/Relu:activations:0)main_output/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp"^aux_output/BiasAdd/ReadVariableOp!^aux_output/MatMul/ReadVariableOp ^dense_78/BiasAdd/ReadVariableOp^dense_78/MatMul/ReadVariableOp ^dense_79/BiasAdd/ReadVariableOp^dense_79/MatMul/ReadVariableOp ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp#^main_output/BiasAdd/ReadVariableOp"^main_output/MatMul/ReadVariableOp=^token_and_position_embedding_8/embedding_16/embedding_lookup=^token_and_position_embedding_8/embedding_17/embedding_lookupD^transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpD^transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpP^transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpR^transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpP^transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpR^transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpP^transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpR^transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpP^transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpR^transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpA^transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOpC^transformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOpA^transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOpC^transformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!aux_output/BiasAdd/ReadVariableOp!aux_output/BiasAdd/ReadVariableOp2D
 aux_output/MatMul/ReadVariableOp aux_output/MatMul/ReadVariableOp2B
dense_78/BiasAdd/ReadVariableOpdense_78/BiasAdd/ReadVariableOp2@
dense_78/MatMul/ReadVariableOpdense_78/MatMul/ReadVariableOp2B
dense_79/BiasAdd/ReadVariableOpdense_79/BiasAdd/ReadVariableOp2@
dense_79/MatMul/ReadVariableOpdense_79/MatMul/ReadVariableOp2B
dense_80/BiasAdd/ReadVariableOpdense_80/BiasAdd/ReadVariableOp2@
dense_80/MatMul/ReadVariableOpdense_80/MatMul/ReadVariableOp2H
"main_output/BiasAdd/ReadVariableOp"main_output/BiasAdd/ReadVariableOp2F
!main_output/MatMul/ReadVariableOp!main_output/MatMul/ReadVariableOp2|
<token_and_position_embedding_8/embedding_16/embedding_lookup<token_and_position_embedding_8/embedding_16/embedding_lookup2|
<token_and_position_embedding_8/embedding_17/embedding_lookup<token_and_position_embedding_8/embedding_17/embedding_lookup2?
Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp2?
Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp2?
Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp2?
Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp2?
Otransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpOtransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp2?
Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp2?
Otransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpOtransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp2?
Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp2?
Otransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpOtransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp2?
Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp2?
Otransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpOtransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp2?
Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp2?
@transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp@transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp2?
Btransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOpBtransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp2?
@transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp@transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp2?
Btransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOpBtransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
5__inference_transformer_block_8_layer_call_fn_4833925

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
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_48320232
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
G__inference_aux_output_layer_call_and_return_conditional_losses_4832075

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
.__inference_sequential_8_layer_call_fn_4831615
dense_76_input
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_76_inputunknown	unknown_0	unknown_1	unknown_2*
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_48316042
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
_user_specified_namedense_76_input
?
?
*__inference_dense_76_layer_call_fn_4834748

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
E__inference_dense_76_layer_call_and_return_conditional_losses_48315612
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
?
?
*__inference_dense_77_layer_call_fn_4834788

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
E__inference_dense_77_layer_call_and_return_conditional_losses_48315972
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
H__inference_main_output_layer_call_and_return_conditional_losses_4832152

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
?
?
@__inference_token_and_position_embedding_8_layer_call_fn_4833864
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
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_48317732
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
? 
?
E__inference_dense_77_layer_call_and_return_conditional_losses_4834818

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
?
?
*__inference_dense_78_layer_call_fn_4834528

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
GPU 2J 8? *N
fIRG
E__inference_dense_78_layer_call_and_return_conditional_losses_48321012
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
?
?
I__inference_sequential_8_layer_call_and_return_conditional_losses_4831664

inputs"
dense_76_4831653:  
dense_76_4831655: "
dense_77_4831658:  
dense_77_4831660: 
identity?? dense_76/StatefulPartitionedCall? dense_77/StatefulPartitionedCall?
 dense_76/StatefulPartitionedCallStatefulPartitionedCallinputsdense_76_4831653dense_76_4831655*
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
E__inference_dense_76_layer_call_and_return_conditional_losses_48315612"
 dense_76/StatefulPartitionedCall?
 dense_77/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0dense_77_4831658dense_77_4831660*
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
E__inference_dense_77_layer_call_and_return_conditional_losses_48315972"
 dense_77/StatefulPartitionedCall?
IdentityIdentity)dense_77/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp!^dense_76/StatefulPartitionedCall!^dense_77/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2D
 dense_77/StatefulPartitionedCall dense_77/StatefulPartitionedCall:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
v
J__inference_concatenate_8_layer_call_and_return_conditional_losses_4834519
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
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
E__inference_dense_80_layer_call_and_return_conditional_losses_4834579

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
?
?
.__inference_sequential_8_layer_call_fn_4834625

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
I__inference_sequential_8_layer_call_and_return_conditional_losses_48316642
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
?<
?
D__inference_model_8_layer_call_and_return_conditional_losses_4832160

inputs
inputs_18
&token_and_position_embedding_8_4831774:( 8
&token_and_position_embedding_8_4831776: -
transformer_block_8_4832024:  )
transformer_block_8_4832026: -
transformer_block_8_4832028:  )
transformer_block_8_4832030: -
transformer_block_8_4832032:  )
transformer_block_8_4832034: -
transformer_block_8_4832036:  )
transformer_block_8_4832038: )
transformer_block_8_4832040: )
transformer_block_8_4832042: -
transformer_block_8_4832044:  )
transformer_block_8_4832046: -
transformer_block_8_4832048:  )
transformer_block_8_4832050: )
transformer_block_8_4832052: )
transformer_block_8_4832054: $
aux_output_4832076:  
aux_output_4832078:"
dense_78_4832102:@
dense_78_4832104:@"
dense_79_4832119:@@
dense_79_4832121:@"
dense_80_4832136:@@
dense_80_4832138:@%
main_output_4832153:@!
main_output_4832155:
identity

identity_1??"aux_output/StatefulPartitionedCall? dense_78/StatefulPartitionedCall? dense_79/StatefulPartitionedCall? dense_80/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?6token_and_position_embedding_8/StatefulPartitionedCall?+transformer_block_8/StatefulPartitionedCall?
6token_and_position_embedding_8/StatefulPartitionedCallStatefulPartitionedCallinputs&token_and_position_embedding_8_4831774&token_and_position_embedding_8_4831776*
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
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_483177328
6token_and_position_embedding_8/StatefulPartitionedCall?
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_8/StatefulPartitionedCall:output:0transformer_block_8_4832024transformer_block_8_4832026transformer_block_8_4832028transformer_block_8_4832030transformer_block_8_4832032transformer_block_8_4832034transformer_block_8_4832036transformer_block_8_4832038transformer_block_8_4832040transformer_block_8_4832042transformer_block_8_4832044transformer_block_8_4832046transformer_block_8_4832048transformer_block_8_4832050transformer_block_8_4832052transformer_block_8_4832054*
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
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_48320232-
+transformer_block_8/StatefulPartitionedCall?
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
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
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_48320622,
*global_average_pooling1d_8/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0aux_output_4832076aux_output_4832078*
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
G__inference_aux_output_layer_call_and_return_conditional_losses_48320752$
"aux_output/StatefulPartitionedCall?
concatenate_8/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
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
GPU 2J 8? *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_48320882
concatenate_8/PartitionedCall?
 dense_78/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_78_4832102dense_78_4832104*
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
E__inference_dense_78_layer_call_and_return_conditional_losses_48321012"
 dense_78/StatefulPartitionedCall?
 dense_79/StatefulPartitionedCallStatefulPartitionedCall)dense_78/StatefulPartitionedCall:output:0dense_79_4832119dense_79_4832121*
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
E__inference_dense_79_layer_call_and_return_conditional_losses_48321182"
 dense_79/StatefulPartitionedCall?
 dense_80/StatefulPartitionedCallStatefulPartitionedCall)dense_79/StatefulPartitionedCall:output:0dense_80_4832136dense_80_4832138*
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
E__inference_dense_80_layer_call_and_return_conditional_losses_48321352"
 dense_80/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0main_output_4832153main_output_4832155*
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
H__inference_main_output_layer_call_and_return_conditional_losses_48321522%
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
NoOpNoOp#^aux_output/StatefulPartitionedCall!^dense_78/StatefulPartitionedCall!^dense_79/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall$^main_output/StatefulPartitionedCall7^token_and_position_embedding_8/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2p
6token_and_position_embedding_8/StatefulPartitionedCall6token_and_position_embedding_8/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_4832581

inputsX
Fmulti_head_self_attention_8_dense_72_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_72_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_73_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_73_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_74_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_74_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_75_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_75_biasadd_readvariableop_resource: J
<layer_normalization_16_batchnorm_mul_readvariableop_resource: F
8layer_normalization_16_batchnorm_readvariableop_resource: I
7sequential_8_dense_76_tensordot_readvariableop_resource:  C
5sequential_8_dense_76_biasadd_readvariableop_resource: I
7sequential_8_dense_77_tensordot_readvariableop_resource:  C
5sequential_8_dense_77_biasadd_readvariableop_resource: J
<layer_normalization_17_batchnorm_mul_readvariableop_resource: F
8layer_normalization_17_batchnorm_readvariableop_resource: 
identity??/layer_normalization_16/batchnorm/ReadVariableOp?3layer_normalization_16/batchnorm/mul/ReadVariableOp?/layer_normalization_17/batchnorm/ReadVariableOp?3layer_normalization_17/batchnorm/mul/ReadVariableOp?;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?,sequential_8/dense_76/BiasAdd/ReadVariableOp?.sequential_8/dense_76/Tensordot/ReadVariableOp?,sequential_8/dense_77/BiasAdd/ReadVariableOp?.sequential_8/dense_77/Tensordot/ReadVariableOp|
!multi_head_self_attention_8/ShapeShapeinputs*
T0*
_output_shapes
:2#
!multi_head_self_attention_8/Shape?
/multi_head_self_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention_8/strided_slice/stack?
1multi_head_self_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_8/strided_slice/stack_1?
1multi_head_self_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_8/strided_slice/stack_2?
)multi_head_self_attention_8/strided_sliceStridedSlice*multi_head_self_attention_8/Shape:output:08multi_head_self_attention_8/strided_slice/stack:output:0:multi_head_self_attention_8/strided_slice/stack_1:output:0:multi_head_self_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention_8/strided_slice?
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_72_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_72/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_72/Tensordot/axes?
3multi_head_self_attention_8/dense_72/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_72/Tensordot/free?
4multi_head_self_attention_8/dense_72/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_72/Tensordot/Shape?
<multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_72/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_72/Tensordot/free:output:0Emulti_head_self_attention_8/dense_72/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_72/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_72/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_72/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_72/Tensordot/Const?
3multi_head_self_attention_8/dense_72/Tensordot/ProdProd@multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_72/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_72/Tensordot/Prod?
6multi_head_self_attention_8/dense_72/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_72/Tensordot/Const_1?
5multi_head_self_attention_8/dense_72/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_72/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_72/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_72/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_72/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_72/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_72/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_72/Tensordot/free:output:0<multi_head_self_attention_8/dense_72/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_72/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_72/Tensordot/concat?
4multi_head_self_attention_8/dense_72/Tensordot/stackPack<multi_head_self_attention_8/dense_72/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_72/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_72/Tensordot/stack?
8multi_head_self_attention_8/dense_72/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_72/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_72/Tensordot/transpose?
6multi_head_self_attention_8/dense_72/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_72/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_72/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_72/Tensordot/Reshape?
5multi_head_self_attention_8/dense_72/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_72/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_72/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_72/Tensordot/MatMul?
6multi_head_self_attention_8/dense_72/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_72/Tensordot/Const_2?
<multi_head_self_attention_8/dense_72/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_72/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_72/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_72/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_72/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_72/Tensordot/concat_1?
.multi_head_self_attention_8/dense_72/TensordotReshape?multi_head_self_attention_8/dense_72/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_72/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_72/Tensordot?
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_72_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_72/BiasAddBiasAdd7multi_head_self_attention_8/dense_72/Tensordot:output:0Cmulti_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_72/BiasAdd?
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_73_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_73/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_73/Tensordot/axes?
3multi_head_self_attention_8/dense_73/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_73/Tensordot/free?
4multi_head_self_attention_8/dense_73/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_73/Tensordot/Shape?
<multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_73/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_73/Tensordot/free:output:0Emulti_head_self_attention_8/dense_73/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_73/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_73/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_73/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_73/Tensordot/Const?
3multi_head_self_attention_8/dense_73/Tensordot/ProdProd@multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_73/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_73/Tensordot/Prod?
6multi_head_self_attention_8/dense_73/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_73/Tensordot/Const_1?
5multi_head_self_attention_8/dense_73/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_73/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_73/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_73/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_73/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_73/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_73/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_73/Tensordot/free:output:0<multi_head_self_attention_8/dense_73/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_73/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_73/Tensordot/concat?
4multi_head_self_attention_8/dense_73/Tensordot/stackPack<multi_head_self_attention_8/dense_73/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_73/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_73/Tensordot/stack?
8multi_head_self_attention_8/dense_73/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_73/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_73/Tensordot/transpose?
6multi_head_self_attention_8/dense_73/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_73/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_73/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_73/Tensordot/Reshape?
5multi_head_self_attention_8/dense_73/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_73/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_73/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_73/Tensordot/MatMul?
6multi_head_self_attention_8/dense_73/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_73/Tensordot/Const_2?
<multi_head_self_attention_8/dense_73/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_73/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_73/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_73/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_73/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_73/Tensordot/concat_1?
.multi_head_self_attention_8/dense_73/TensordotReshape?multi_head_self_attention_8/dense_73/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_73/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_73/Tensordot?
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_73/BiasAddBiasAdd7multi_head_self_attention_8/dense_73/Tensordot:output:0Cmulti_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_73/BiasAdd?
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_74_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_74/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_74/Tensordot/axes?
3multi_head_self_attention_8/dense_74/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_74/Tensordot/free?
4multi_head_self_attention_8/dense_74/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_74/Tensordot/Shape?
<multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_74/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_74/Tensordot/free:output:0Emulti_head_self_attention_8/dense_74/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_74/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_74/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_74/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_74/Tensordot/Const?
3multi_head_self_attention_8/dense_74/Tensordot/ProdProd@multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_74/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_74/Tensordot/Prod?
6multi_head_self_attention_8/dense_74/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_74/Tensordot/Const_1?
5multi_head_self_attention_8/dense_74/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_74/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_74/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_74/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_74/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_74/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_74/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_74/Tensordot/free:output:0<multi_head_self_attention_8/dense_74/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_74/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_74/Tensordot/concat?
4multi_head_self_attention_8/dense_74/Tensordot/stackPack<multi_head_self_attention_8/dense_74/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_74/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_74/Tensordot/stack?
8multi_head_self_attention_8/dense_74/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_74/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_74/Tensordot/transpose?
6multi_head_self_attention_8/dense_74/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_74/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_74/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_74/Tensordot/Reshape?
5multi_head_self_attention_8/dense_74/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_74/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_74/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_74/Tensordot/MatMul?
6multi_head_self_attention_8/dense_74/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_74/Tensordot/Const_2?
<multi_head_self_attention_8/dense_74/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_74/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_74/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_74/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_74/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_74/Tensordot/concat_1?
.multi_head_self_attention_8/dense_74/TensordotReshape?multi_head_self_attention_8/dense_74/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_74/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_74/Tensordot?
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_74/BiasAddBiasAdd7multi_head_self_attention_8/dense_74/Tensordot:output:0Cmulti_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_74/BiasAdd?
+multi_head_self_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+multi_head_self_attention_8/Reshape/shape/1?
+multi_head_self_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_8/Reshape/shape/2?
+multi_head_self_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_8/Reshape/shape/3?
)multi_head_self_attention_8/Reshape/shapePack2multi_head_self_attention_8/strided_slice:output:04multi_head_self_attention_8/Reshape/shape/1:output:04multi_head_self_attention_8/Reshape/shape/2:output:04multi_head_self_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention_8/Reshape/shape?
#multi_head_self_attention_8/ReshapeReshape5multi_head_self_attention_8/dense_72/BiasAdd:output:02multi_head_self_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2%
#multi_head_self_attention_8/Reshape?
*multi_head_self_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention_8/transpose/perm?
%multi_head_self_attention_8/transpose	Transpose,multi_head_self_attention_8/Reshape:output:03multi_head_self_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/transpose?
-multi_head_self_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_1/shape/1?
-multi_head_self_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_1/shape/2?
-multi_head_self_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_1/shape/3?
+multi_head_self_attention_8/Reshape_1/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_1/shape/1:output:06multi_head_self_attention_8/Reshape_1/shape/2:output:06multi_head_self_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_1/shape?
%multi_head_self_attention_8/Reshape_1Reshape5multi_head_self_attention_8/dense_73/BiasAdd:output:04multi_head_self_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/Reshape_1?
,multi_head_self_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_1/perm?
'multi_head_self_attention_8/transpose_1	Transpose.multi_head_self_attention_8/Reshape_1:output:05multi_head_self_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_1?
-multi_head_self_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_2/shape/1?
-multi_head_self_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_2/shape/2?
-multi_head_self_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_2/shape/3?
+multi_head_self_attention_8/Reshape_2/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_2/shape/1:output:06multi_head_self_attention_8/Reshape_2/shape/2:output:06multi_head_self_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_2/shape?
%multi_head_self_attention_8/Reshape_2Reshape5multi_head_self_attention_8/dense_74/BiasAdd:output:04multi_head_self_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/Reshape_2?
,multi_head_self_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_2/perm?
'multi_head_self_attention_8/transpose_2	Transpose.multi_head_self_attention_8/Reshape_2:output:05multi_head_self_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_2?
"multi_head_self_attention_8/MatMulBatchMatMulV2)multi_head_self_attention_8/transpose:y:0+multi_head_self_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2$
"multi_head_self_attention_8/MatMul?
#multi_head_self_attention_8/Shape_1Shape+multi_head_self_attention_8/transpose_1:y:0*
T0*
_output_shapes
:2%
#multi_head_self_attention_8/Shape_1?
1multi_head_self_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1multi_head_self_attention_8/strided_slice_1/stack?
3multi_head_self_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention_8/strided_slice_1/stack_1?
3multi_head_self_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/strided_slice_1/stack_2?
+multi_head_self_attention_8/strided_slice_1StridedSlice,multi_head_self_attention_8/Shape_1:output:0:multi_head_self_attention_8/strided_slice_1/stack:output:0<multi_head_self_attention_8/strided_slice_1/stack_1:output:0<multi_head_self_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+multi_head_self_attention_8/strided_slice_1?
 multi_head_self_attention_8/CastCast4multi_head_self_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 multi_head_self_attention_8/Cast?
 multi_head_self_attention_8/SqrtSqrt$multi_head_self_attention_8/Cast:y:0*
T0*
_output_shapes
: 2"
 multi_head_self_attention_8/Sqrt?
#multi_head_self_attention_8/truedivRealDiv+multi_head_self_attention_8/MatMul:output:0$multi_head_self_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_8/truediv?
#multi_head_self_attention_8/SoftmaxSoftmax'multi_head_self_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_8/Softmax?
$multi_head_self_attention_8/MatMul_1BatchMatMulV2-multi_head_self_attention_8/Softmax:softmax:0+multi_head_self_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_8/MatMul_1?
,multi_head_self_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_3/perm?
'multi_head_self_attention_8/transpose_3	Transpose-multi_head_self_attention_8/MatMul_1:output:05multi_head_self_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_3?
-multi_head_self_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_3/shape/1?
-multi_head_self_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2/
-multi_head_self_attention_8/Reshape_3/shape/2?
+multi_head_self_attention_8/Reshape_3/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_3/shape/1:output:06multi_head_self_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_3/shape?
%multi_head_self_attention_8/Reshape_3Reshape+multi_head_self_attention_8/transpose_3:y:04multi_head_self_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2'
%multi_head_self_attention_8/Reshape_3?
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_75_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_75/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_75/Tensordot/axes?
3multi_head_self_attention_8/dense_75/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_75/Tensordot/free?
4multi_head_self_attention_8/dense_75/Tensordot/ShapeShape.multi_head_self_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_75/Tensordot/Shape?
<multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_75/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_75/Tensordot/free:output:0Emulti_head_self_attention_8/dense_75/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_75/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_75/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_75/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_75/Tensordot/Const?
3multi_head_self_attention_8/dense_75/Tensordot/ProdProd@multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_75/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_75/Tensordot/Prod?
6multi_head_self_attention_8/dense_75/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_75/Tensordot/Const_1?
5multi_head_self_attention_8/dense_75/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_75/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_75/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_75/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_75/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_75/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_75/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_75/Tensordot/free:output:0<multi_head_self_attention_8/dense_75/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_75/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_75/Tensordot/concat?
4multi_head_self_attention_8/dense_75/Tensordot/stackPack<multi_head_self_attention_8/dense_75/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_75/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_75/Tensordot/stack?
8multi_head_self_attention_8/dense_75/Tensordot/transpose	Transpose.multi_head_self_attention_8/Reshape_3:output:0>multi_head_self_attention_8/dense_75/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2:
8multi_head_self_attention_8/dense_75/Tensordot/transpose?
6multi_head_self_attention_8/dense_75/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_75/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_75/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_75/Tensordot/Reshape?
5multi_head_self_attention_8/dense_75/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_75/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_75/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_75/Tensordot/MatMul?
6multi_head_self_attention_8/dense_75/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_75/Tensordot/Const_2?
<multi_head_self_attention_8/dense_75/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_75/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_75/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_75/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_75/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_75/Tensordot/concat_1?
.multi_head_self_attention_8/dense_75/TensordotReshape?multi_head_self_attention_8/dense_75/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_75/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_8/dense_75/Tensordot?
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_75_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_75/BiasAddBiasAdd7multi_head_self_attention_8/dense_75/Tensordot:output:0Cmulti_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2.
,multi_head_self_attention_8/dense_75/BiasAddy
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_16/dropout/Const?
dropout_16/dropout/MulMul5multi_head_self_attention_8/dense_75/BiasAdd:output:0!dropout_16/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_16/dropout/Mul?
dropout_16/dropout/ShapeShape5multi_head_self_attention_8/dense_75/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype021
/dropout_16/dropout/random_uniform/RandomUniform?
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_16/dropout/GreaterEqual/y?
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2!
dropout_16/dropout/GreaterEqual?
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout_16/dropout/Cast?
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_16/dropout/Mul_1o
addAddV2inputsdropout_16/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add?
5layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_16/moments/mean/reduction_indices?
#layer_normalization_16/moments/meanMeanadd:z:0>layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_16/moments/mean?
+layer_normalization_16/moments/StopGradientStopGradient,layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_16/moments/StopGradient?
0layer_normalization_16/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_16/moments/SquaredDifference?
9layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_16/moments/variance/reduction_indices?
'layer_normalization_16/moments/varianceMean4layer_normalization_16/moments/SquaredDifference:z:0Blayer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_16/moments/variance?
&layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_16/batchnorm/add/y?
$layer_normalization_16/batchnorm/addAddV20layer_normalization_16/moments/variance:output:0/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_16/batchnorm/add?
&layer_normalization_16/batchnorm/RsqrtRsqrt(layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_16/batchnorm/Rsqrt?
3layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_16/batchnorm/mul/ReadVariableOp?
$layer_normalization_16/batchnorm/mulMul*layer_normalization_16/batchnorm/Rsqrt:y:0;layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_16/batchnorm/mul?
&layer_normalization_16/batchnorm/mul_1Muladd:z:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/mul_1?
&layer_normalization_16/batchnorm/mul_2Mul,layer_normalization_16/moments/mean:output:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/mul_2?
/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_16/batchnorm/ReadVariableOp?
$layer_normalization_16/batchnorm/subSub7layer_normalization_16/batchnorm/ReadVariableOp:value:0*layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_16/batchnorm/sub?
&layer_normalization_16/batchnorm/add_1AddV2*layer_normalization_16/batchnorm/mul_1:z:0(layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/add_1?
.sequential_8/dense_76/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_76_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_8/dense_76/Tensordot/ReadVariableOp?
$sequential_8/dense_76/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_8/dense_76/Tensordot/axes?
$sequential_8/dense_76/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_8/dense_76/Tensordot/free?
%sequential_8/dense_76/Tensordot/ShapeShape*layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_8/dense_76/Tensordot/Shape?
-sequential_8/dense_76/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_76/Tensordot/GatherV2/axis?
(sequential_8/dense_76/Tensordot/GatherV2GatherV2.sequential_8/dense_76/Tensordot/Shape:output:0-sequential_8/dense_76/Tensordot/free:output:06sequential_8/dense_76/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_8/dense_76/Tensordot/GatherV2?
/sequential_8/dense_76/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_8/dense_76/Tensordot/GatherV2_1/axis?
*sequential_8/dense_76/Tensordot/GatherV2_1GatherV2.sequential_8/dense_76/Tensordot/Shape:output:0-sequential_8/dense_76/Tensordot/axes:output:08sequential_8/dense_76/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_8/dense_76/Tensordot/GatherV2_1?
%sequential_8/dense_76/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_8/dense_76/Tensordot/Const?
$sequential_8/dense_76/Tensordot/ProdProd1sequential_8/dense_76/Tensordot/GatherV2:output:0.sequential_8/dense_76/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_8/dense_76/Tensordot/Prod?
'sequential_8/dense_76/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_76/Tensordot/Const_1?
&sequential_8/dense_76/Tensordot/Prod_1Prod3sequential_8/dense_76/Tensordot/GatherV2_1:output:00sequential_8/dense_76/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_8/dense_76/Tensordot/Prod_1?
+sequential_8/dense_76/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_8/dense_76/Tensordot/concat/axis?
&sequential_8/dense_76/Tensordot/concatConcatV2-sequential_8/dense_76/Tensordot/free:output:0-sequential_8/dense_76/Tensordot/axes:output:04sequential_8/dense_76/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_76/Tensordot/concat?
%sequential_8/dense_76/Tensordot/stackPack-sequential_8/dense_76/Tensordot/Prod:output:0/sequential_8/dense_76/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/dense_76/Tensordot/stack?
)sequential_8/dense_76/Tensordot/transpose	Transpose*layer_normalization_16/batchnorm/add_1:z:0/sequential_8/dense_76/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_8/dense_76/Tensordot/transpose?
'sequential_8/dense_76/Tensordot/ReshapeReshape-sequential_8/dense_76/Tensordot/transpose:y:0.sequential_8/dense_76/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_8/dense_76/Tensordot/Reshape?
&sequential_8/dense_76/Tensordot/MatMulMatMul0sequential_8/dense_76/Tensordot/Reshape:output:06sequential_8/dense_76/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_8/dense_76/Tensordot/MatMul?
'sequential_8/dense_76/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_76/Tensordot/Const_2?
-sequential_8/dense_76/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_76/Tensordot/concat_1/axis?
(sequential_8/dense_76/Tensordot/concat_1ConcatV21sequential_8/dense_76/Tensordot/GatherV2:output:00sequential_8/dense_76/Tensordot/Const_2:output:06sequential_8/dense_76/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_8/dense_76/Tensordot/concat_1?
sequential_8/dense_76/TensordotReshape0sequential_8/dense_76/Tensordot/MatMul:product:01sequential_8/dense_76/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_8/dense_76/Tensordot?
,sequential_8/dense_76/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_76_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_8/dense_76/BiasAdd/ReadVariableOp?
sequential_8/dense_76/BiasAddBiasAdd(sequential_8/dense_76/Tensordot:output:04sequential_8/dense_76/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_76/BiasAdd?
sequential_8/dense_76/ReluRelu&sequential_8/dense_76/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_76/Relu?
.sequential_8/dense_77/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_77_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_8/dense_77/Tensordot/ReadVariableOp?
$sequential_8/dense_77/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_8/dense_77/Tensordot/axes?
$sequential_8/dense_77/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_8/dense_77/Tensordot/free?
%sequential_8/dense_77/Tensordot/ShapeShape(sequential_8/dense_76/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_8/dense_77/Tensordot/Shape?
-sequential_8/dense_77/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_77/Tensordot/GatherV2/axis?
(sequential_8/dense_77/Tensordot/GatherV2GatherV2.sequential_8/dense_77/Tensordot/Shape:output:0-sequential_8/dense_77/Tensordot/free:output:06sequential_8/dense_77/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_8/dense_77/Tensordot/GatherV2?
/sequential_8/dense_77/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_8/dense_77/Tensordot/GatherV2_1/axis?
*sequential_8/dense_77/Tensordot/GatherV2_1GatherV2.sequential_8/dense_77/Tensordot/Shape:output:0-sequential_8/dense_77/Tensordot/axes:output:08sequential_8/dense_77/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_8/dense_77/Tensordot/GatherV2_1?
%sequential_8/dense_77/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_8/dense_77/Tensordot/Const?
$sequential_8/dense_77/Tensordot/ProdProd1sequential_8/dense_77/Tensordot/GatherV2:output:0.sequential_8/dense_77/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_8/dense_77/Tensordot/Prod?
'sequential_8/dense_77/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_77/Tensordot/Const_1?
&sequential_8/dense_77/Tensordot/Prod_1Prod3sequential_8/dense_77/Tensordot/GatherV2_1:output:00sequential_8/dense_77/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_8/dense_77/Tensordot/Prod_1?
+sequential_8/dense_77/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_8/dense_77/Tensordot/concat/axis?
&sequential_8/dense_77/Tensordot/concatConcatV2-sequential_8/dense_77/Tensordot/free:output:0-sequential_8/dense_77/Tensordot/axes:output:04sequential_8/dense_77/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_77/Tensordot/concat?
%sequential_8/dense_77/Tensordot/stackPack-sequential_8/dense_77/Tensordot/Prod:output:0/sequential_8/dense_77/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/dense_77/Tensordot/stack?
)sequential_8/dense_77/Tensordot/transpose	Transpose(sequential_8/dense_76/Relu:activations:0/sequential_8/dense_77/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_8/dense_77/Tensordot/transpose?
'sequential_8/dense_77/Tensordot/ReshapeReshape-sequential_8/dense_77/Tensordot/transpose:y:0.sequential_8/dense_77/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_8/dense_77/Tensordot/Reshape?
&sequential_8/dense_77/Tensordot/MatMulMatMul0sequential_8/dense_77/Tensordot/Reshape:output:06sequential_8/dense_77/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_8/dense_77/Tensordot/MatMul?
'sequential_8/dense_77/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_77/Tensordot/Const_2?
-sequential_8/dense_77/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_77/Tensordot/concat_1/axis?
(sequential_8/dense_77/Tensordot/concat_1ConcatV21sequential_8/dense_77/Tensordot/GatherV2:output:00sequential_8/dense_77/Tensordot/Const_2:output:06sequential_8/dense_77/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_8/dense_77/Tensordot/concat_1?
sequential_8/dense_77/TensordotReshape0sequential_8/dense_77/Tensordot/MatMul:product:01sequential_8/dense_77/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_8/dense_77/Tensordot?
,sequential_8/dense_77/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_8/dense_77/BiasAdd/ReadVariableOp?
sequential_8/dense_77/BiasAddBiasAdd(sequential_8/dense_77/Tensordot:output:04sequential_8/dense_77/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_77/BiasAddy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_17/dropout/Const?
dropout_17/dropout/MulMul&sequential_8/dense_77/BiasAdd:output:0!dropout_17/dropout/Const:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_17/dropout/Mul?
dropout_17/dropout/ShapeShape&sequential_8/dense_77/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape?
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????( *
dtype021
/dropout_17/dropout/random_uniform/RandomUniform?
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_17/dropout/GreaterEqual/y?
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????( 2!
dropout_17/dropout/GreaterEqual?
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????( 2
dropout_17/dropout/Cast?
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????( 2
dropout_17/dropout/Mul_1?
add_1AddV2*layer_normalization_16/batchnorm/add_1:z:0dropout_17/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add_1?
5layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_17/moments/mean/reduction_indices?
#layer_normalization_17/moments/meanMean	add_1:z:0>layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_17/moments/mean?
+layer_normalization_17/moments/StopGradientStopGradient,layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_17/moments/StopGradient?
0layer_normalization_17/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_17/moments/SquaredDifference?
9layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_17/moments/variance/reduction_indices?
'layer_normalization_17/moments/varianceMean4layer_normalization_17/moments/SquaredDifference:z:0Blayer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_17/moments/variance?
&layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_17/batchnorm/add/y?
$layer_normalization_17/batchnorm/addAddV20layer_normalization_17/moments/variance:output:0/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_17/batchnorm/add?
&layer_normalization_17/batchnorm/RsqrtRsqrt(layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_17/batchnorm/Rsqrt?
3layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_17/batchnorm/mul/ReadVariableOp?
$layer_normalization_17/batchnorm/mulMul*layer_normalization_17/batchnorm/Rsqrt:y:0;layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_17/batchnorm/mul?
&layer_normalization_17/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/mul_1?
&layer_normalization_17/batchnorm/mul_2Mul,layer_normalization_17/moments/mean:output:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/mul_2?
/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_17/batchnorm/ReadVariableOp?
$layer_normalization_17/batchnorm/subSub7layer_normalization_17/batchnorm/ReadVariableOp:value:0*layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_17/batchnorm/sub?
&layer_normalization_17/batchnorm/add_1AddV2*layer_normalization_17/batchnorm/mul_1:z:0(layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/add_1?
IdentityIdentity*layer_normalization_17/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp0^layer_normalization_16/batchnorm/ReadVariableOp4^layer_normalization_16/batchnorm/mul/ReadVariableOp0^layer_normalization_17/batchnorm/ReadVariableOp4^layer_normalization_17/batchnorm/mul/ReadVariableOp<^multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp-^sequential_8/dense_76/BiasAdd/ReadVariableOp/^sequential_8/dense_76/Tensordot/ReadVariableOp-^sequential_8/dense_77/BiasAdd/ReadVariableOp/^sequential_8/dense_77/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2b
/layer_normalization_16/batchnorm/ReadVariableOp/layer_normalization_16/batchnorm/ReadVariableOp2j
3layer_normalization_16/batchnorm/mul/ReadVariableOp3layer_normalization_16/batchnorm/mul/ReadVariableOp2b
/layer_normalization_17/batchnorm/ReadVariableOp/layer_normalization_17/batchnorm/ReadVariableOp2j
3layer_normalization_17/batchnorm/mul/ReadVariableOp3layer_normalization_17/batchnorm/mul/ReadVariableOp2z
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp2\
,sequential_8/dense_76/BiasAdd/ReadVariableOp,sequential_8/dense_76/BiasAdd/ReadVariableOp2`
.sequential_8/dense_76/Tensordot/ReadVariableOp.sequential_8/dense_76/Tensordot/ReadVariableOp2\
,sequential_8/dense_77/BiasAdd/ReadVariableOp,sequential_8/dense_77/BiasAdd/ReadVariableOp2`
.sequential_8/dense_77/Tensordot/ReadVariableOp.sequential_8/dense_77/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?<
?
D__inference_model_8_layer_call_and_return_conditional_losses_4832960
input_9
	aux_input8
&token_and_position_embedding_8_4832893:( 8
&token_and_position_embedding_8_4832895: -
transformer_block_8_4832898:  )
transformer_block_8_4832900: -
transformer_block_8_4832902:  )
transformer_block_8_4832904: -
transformer_block_8_4832906:  )
transformer_block_8_4832908: -
transformer_block_8_4832910:  )
transformer_block_8_4832912: )
transformer_block_8_4832914: )
transformer_block_8_4832916: -
transformer_block_8_4832918:  )
transformer_block_8_4832920: -
transformer_block_8_4832922:  )
transformer_block_8_4832924: )
transformer_block_8_4832926: )
transformer_block_8_4832928: $
aux_output_4832932:  
aux_output_4832934:"
dense_78_4832938:@
dense_78_4832940:@"
dense_79_4832943:@@
dense_79_4832945:@"
dense_80_4832948:@@
dense_80_4832950:@%
main_output_4832953:@!
main_output_4832955:
identity

identity_1??"aux_output/StatefulPartitionedCall? dense_78/StatefulPartitionedCall? dense_79/StatefulPartitionedCall? dense_80/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?6token_and_position_embedding_8/StatefulPartitionedCall?+transformer_block_8/StatefulPartitionedCall?
6token_and_position_embedding_8/StatefulPartitionedCallStatefulPartitionedCallinput_9&token_and_position_embedding_8_4832893&token_and_position_embedding_8_4832895*
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
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_483177328
6token_and_position_embedding_8/StatefulPartitionedCall?
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_8/StatefulPartitionedCall:output:0transformer_block_8_4832898transformer_block_8_4832900transformer_block_8_4832902transformer_block_8_4832904transformer_block_8_4832906transformer_block_8_4832908transformer_block_8_4832910transformer_block_8_4832912transformer_block_8_4832914transformer_block_8_4832916transformer_block_8_4832918transformer_block_8_4832920transformer_block_8_4832922transformer_block_8_4832924transformer_block_8_4832926transformer_block_8_4832928*
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
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_48320232-
+transformer_block_8/StatefulPartitionedCall?
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
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
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_48320622,
*global_average_pooling1d_8/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0aux_output_4832932aux_output_4832934*
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
G__inference_aux_output_layer_call_and_return_conditional_losses_48320752$
"aux_output/StatefulPartitionedCall?
concatenate_8/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0	aux_input*
Tin
2*
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
GPU 2J 8? *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_48320882
concatenate_8/PartitionedCall?
 dense_78/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_78_4832938dense_78_4832940*
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
E__inference_dense_78_layer_call_and_return_conditional_losses_48321012"
 dense_78/StatefulPartitionedCall?
 dense_79/StatefulPartitionedCallStatefulPartitionedCall)dense_78/StatefulPartitionedCall:output:0dense_79_4832943dense_79_4832945*
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
E__inference_dense_79_layer_call_and_return_conditional_losses_48321182"
 dense_79/StatefulPartitionedCall?
 dense_80/StatefulPartitionedCallStatefulPartitionedCall)dense_79/StatefulPartitionedCall:output:0dense_80_4832948dense_80_4832950*
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
E__inference_dense_80_layer_call_and_return_conditional_losses_48321352"
 dense_80/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0main_output_4832953main_output_4832955*
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
H__inference_main_output_layer_call_and_return_conditional_losses_48321522%
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
NoOpNoOp#^aux_output/StatefulPartitionedCall!^dense_78/StatefulPartitionedCall!^dense_79/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall$^main_output/StatefulPartitionedCall7^token_and_position_embedding_8/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2p
6token_and_position_embedding_8/StatefulPartitionedCall6token_and_position_embedding_8/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:P L
'
_output_shapes
:?????????(
!
_user_specified_name	input_9:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input
? 
?
E__inference_dense_77_layer_call_and_return_conditional_losses_4831597

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
??
?
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_4832023

inputsX
Fmulti_head_self_attention_8_dense_72_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_72_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_73_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_73_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_74_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_74_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_75_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_75_biasadd_readvariableop_resource: J
<layer_normalization_16_batchnorm_mul_readvariableop_resource: F
8layer_normalization_16_batchnorm_readvariableop_resource: I
7sequential_8_dense_76_tensordot_readvariableop_resource:  C
5sequential_8_dense_76_biasadd_readvariableop_resource: I
7sequential_8_dense_77_tensordot_readvariableop_resource:  C
5sequential_8_dense_77_biasadd_readvariableop_resource: J
<layer_normalization_17_batchnorm_mul_readvariableop_resource: F
8layer_normalization_17_batchnorm_readvariableop_resource: 
identity??/layer_normalization_16/batchnorm/ReadVariableOp?3layer_normalization_16/batchnorm/mul/ReadVariableOp?/layer_normalization_17/batchnorm/ReadVariableOp?3layer_normalization_17/batchnorm/mul/ReadVariableOp?;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?,sequential_8/dense_76/BiasAdd/ReadVariableOp?.sequential_8/dense_76/Tensordot/ReadVariableOp?,sequential_8/dense_77/BiasAdd/ReadVariableOp?.sequential_8/dense_77/Tensordot/ReadVariableOp|
!multi_head_self_attention_8/ShapeShapeinputs*
T0*
_output_shapes
:2#
!multi_head_self_attention_8/Shape?
/multi_head_self_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention_8/strided_slice/stack?
1multi_head_self_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_8/strided_slice/stack_1?
1multi_head_self_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_8/strided_slice/stack_2?
)multi_head_self_attention_8/strided_sliceStridedSlice*multi_head_self_attention_8/Shape:output:08multi_head_self_attention_8/strided_slice/stack:output:0:multi_head_self_attention_8/strided_slice/stack_1:output:0:multi_head_self_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention_8/strided_slice?
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_72_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_72/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_72/Tensordot/axes?
3multi_head_self_attention_8/dense_72/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_72/Tensordot/free?
4multi_head_self_attention_8/dense_72/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_72/Tensordot/Shape?
<multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_72/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_72/Tensordot/free:output:0Emulti_head_self_attention_8/dense_72/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_72/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_72/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_72/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_72/Tensordot/Const?
3multi_head_self_attention_8/dense_72/Tensordot/ProdProd@multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_72/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_72/Tensordot/Prod?
6multi_head_self_attention_8/dense_72/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_72/Tensordot/Const_1?
5multi_head_self_attention_8/dense_72/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_72/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_72/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_72/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_72/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_72/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_72/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_72/Tensordot/free:output:0<multi_head_self_attention_8/dense_72/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_72/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_72/Tensordot/concat?
4multi_head_self_attention_8/dense_72/Tensordot/stackPack<multi_head_self_attention_8/dense_72/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_72/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_72/Tensordot/stack?
8multi_head_self_attention_8/dense_72/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_72/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_72/Tensordot/transpose?
6multi_head_self_attention_8/dense_72/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_72/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_72/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_72/Tensordot/Reshape?
5multi_head_self_attention_8/dense_72/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_72/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_72/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_72/Tensordot/MatMul?
6multi_head_self_attention_8/dense_72/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_72/Tensordot/Const_2?
<multi_head_self_attention_8/dense_72/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_72/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_72/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_72/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_72/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_72/Tensordot/concat_1?
.multi_head_self_attention_8/dense_72/TensordotReshape?multi_head_self_attention_8/dense_72/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_72/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_72/Tensordot?
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_72_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_72/BiasAddBiasAdd7multi_head_self_attention_8/dense_72/Tensordot:output:0Cmulti_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_72/BiasAdd?
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_73_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_73/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_73/Tensordot/axes?
3multi_head_self_attention_8/dense_73/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_73/Tensordot/free?
4multi_head_self_attention_8/dense_73/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_73/Tensordot/Shape?
<multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_73/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_73/Tensordot/free:output:0Emulti_head_self_attention_8/dense_73/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_73/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_73/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_73/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_73/Tensordot/Const?
3multi_head_self_attention_8/dense_73/Tensordot/ProdProd@multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_73/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_73/Tensordot/Prod?
6multi_head_self_attention_8/dense_73/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_73/Tensordot/Const_1?
5multi_head_self_attention_8/dense_73/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_73/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_73/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_73/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_73/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_73/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_73/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_73/Tensordot/free:output:0<multi_head_self_attention_8/dense_73/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_73/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_73/Tensordot/concat?
4multi_head_self_attention_8/dense_73/Tensordot/stackPack<multi_head_self_attention_8/dense_73/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_73/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_73/Tensordot/stack?
8multi_head_self_attention_8/dense_73/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_73/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_73/Tensordot/transpose?
6multi_head_self_attention_8/dense_73/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_73/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_73/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_73/Tensordot/Reshape?
5multi_head_self_attention_8/dense_73/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_73/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_73/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_73/Tensordot/MatMul?
6multi_head_self_attention_8/dense_73/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_73/Tensordot/Const_2?
<multi_head_self_attention_8/dense_73/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_73/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_73/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_73/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_73/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_73/Tensordot/concat_1?
.multi_head_self_attention_8/dense_73/TensordotReshape?multi_head_self_attention_8/dense_73/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_73/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_73/Tensordot?
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_73/BiasAddBiasAdd7multi_head_self_attention_8/dense_73/Tensordot:output:0Cmulti_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_73/BiasAdd?
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_74_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_74/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_74/Tensordot/axes?
3multi_head_self_attention_8/dense_74/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_74/Tensordot/free?
4multi_head_self_attention_8/dense_74/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_74/Tensordot/Shape?
<multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_74/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_74/Tensordot/free:output:0Emulti_head_self_attention_8/dense_74/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_74/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_74/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_74/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_74/Tensordot/Const?
3multi_head_self_attention_8/dense_74/Tensordot/ProdProd@multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_74/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_74/Tensordot/Prod?
6multi_head_self_attention_8/dense_74/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_74/Tensordot/Const_1?
5multi_head_self_attention_8/dense_74/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_74/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_74/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_74/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_74/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_74/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_74/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_74/Tensordot/free:output:0<multi_head_self_attention_8/dense_74/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_74/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_74/Tensordot/concat?
4multi_head_self_attention_8/dense_74/Tensordot/stackPack<multi_head_self_attention_8/dense_74/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_74/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_74/Tensordot/stack?
8multi_head_self_attention_8/dense_74/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_74/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_74/Tensordot/transpose?
6multi_head_self_attention_8/dense_74/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_74/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_74/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_74/Tensordot/Reshape?
5multi_head_self_attention_8/dense_74/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_74/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_74/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_74/Tensordot/MatMul?
6multi_head_self_attention_8/dense_74/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_74/Tensordot/Const_2?
<multi_head_self_attention_8/dense_74/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_74/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_74/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_74/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_74/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_74/Tensordot/concat_1?
.multi_head_self_attention_8/dense_74/TensordotReshape?multi_head_self_attention_8/dense_74/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_74/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_74/Tensordot?
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_74/BiasAddBiasAdd7multi_head_self_attention_8/dense_74/Tensordot:output:0Cmulti_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_74/BiasAdd?
+multi_head_self_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+multi_head_self_attention_8/Reshape/shape/1?
+multi_head_self_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_8/Reshape/shape/2?
+multi_head_self_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_8/Reshape/shape/3?
)multi_head_self_attention_8/Reshape/shapePack2multi_head_self_attention_8/strided_slice:output:04multi_head_self_attention_8/Reshape/shape/1:output:04multi_head_self_attention_8/Reshape/shape/2:output:04multi_head_self_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention_8/Reshape/shape?
#multi_head_self_attention_8/ReshapeReshape5multi_head_self_attention_8/dense_72/BiasAdd:output:02multi_head_self_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2%
#multi_head_self_attention_8/Reshape?
*multi_head_self_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention_8/transpose/perm?
%multi_head_self_attention_8/transpose	Transpose,multi_head_self_attention_8/Reshape:output:03multi_head_self_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/transpose?
-multi_head_self_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_1/shape/1?
-multi_head_self_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_1/shape/2?
-multi_head_self_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_1/shape/3?
+multi_head_self_attention_8/Reshape_1/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_1/shape/1:output:06multi_head_self_attention_8/Reshape_1/shape/2:output:06multi_head_self_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_1/shape?
%multi_head_self_attention_8/Reshape_1Reshape5multi_head_self_attention_8/dense_73/BiasAdd:output:04multi_head_self_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/Reshape_1?
,multi_head_self_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_1/perm?
'multi_head_self_attention_8/transpose_1	Transpose.multi_head_self_attention_8/Reshape_1:output:05multi_head_self_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_1?
-multi_head_self_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_2/shape/1?
-multi_head_self_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_2/shape/2?
-multi_head_self_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_2/shape/3?
+multi_head_self_attention_8/Reshape_2/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_2/shape/1:output:06multi_head_self_attention_8/Reshape_2/shape/2:output:06multi_head_self_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_2/shape?
%multi_head_self_attention_8/Reshape_2Reshape5multi_head_self_attention_8/dense_74/BiasAdd:output:04multi_head_self_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/Reshape_2?
,multi_head_self_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_2/perm?
'multi_head_self_attention_8/transpose_2	Transpose.multi_head_self_attention_8/Reshape_2:output:05multi_head_self_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_2?
"multi_head_self_attention_8/MatMulBatchMatMulV2)multi_head_self_attention_8/transpose:y:0+multi_head_self_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2$
"multi_head_self_attention_8/MatMul?
#multi_head_self_attention_8/Shape_1Shape+multi_head_self_attention_8/transpose_1:y:0*
T0*
_output_shapes
:2%
#multi_head_self_attention_8/Shape_1?
1multi_head_self_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1multi_head_self_attention_8/strided_slice_1/stack?
3multi_head_self_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention_8/strided_slice_1/stack_1?
3multi_head_self_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/strided_slice_1/stack_2?
+multi_head_self_attention_8/strided_slice_1StridedSlice,multi_head_self_attention_8/Shape_1:output:0:multi_head_self_attention_8/strided_slice_1/stack:output:0<multi_head_self_attention_8/strided_slice_1/stack_1:output:0<multi_head_self_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+multi_head_self_attention_8/strided_slice_1?
 multi_head_self_attention_8/CastCast4multi_head_self_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 multi_head_self_attention_8/Cast?
 multi_head_self_attention_8/SqrtSqrt$multi_head_self_attention_8/Cast:y:0*
T0*
_output_shapes
: 2"
 multi_head_self_attention_8/Sqrt?
#multi_head_self_attention_8/truedivRealDiv+multi_head_self_attention_8/MatMul:output:0$multi_head_self_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_8/truediv?
#multi_head_self_attention_8/SoftmaxSoftmax'multi_head_self_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_8/Softmax?
$multi_head_self_attention_8/MatMul_1BatchMatMulV2-multi_head_self_attention_8/Softmax:softmax:0+multi_head_self_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_8/MatMul_1?
,multi_head_self_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_3/perm?
'multi_head_self_attention_8/transpose_3	Transpose-multi_head_self_attention_8/MatMul_1:output:05multi_head_self_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_3?
-multi_head_self_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_3/shape/1?
-multi_head_self_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2/
-multi_head_self_attention_8/Reshape_3/shape/2?
+multi_head_self_attention_8/Reshape_3/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_3/shape/1:output:06multi_head_self_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_3/shape?
%multi_head_self_attention_8/Reshape_3Reshape+multi_head_self_attention_8/transpose_3:y:04multi_head_self_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2'
%multi_head_self_attention_8/Reshape_3?
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_75_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_75/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_75/Tensordot/axes?
3multi_head_self_attention_8/dense_75/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_75/Tensordot/free?
4multi_head_self_attention_8/dense_75/Tensordot/ShapeShape.multi_head_self_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_75/Tensordot/Shape?
<multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_75/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_75/Tensordot/free:output:0Emulti_head_self_attention_8/dense_75/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_75/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_75/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_75/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_75/Tensordot/Const?
3multi_head_self_attention_8/dense_75/Tensordot/ProdProd@multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_75/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_75/Tensordot/Prod?
6multi_head_self_attention_8/dense_75/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_75/Tensordot/Const_1?
5multi_head_self_attention_8/dense_75/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_75/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_75/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_75/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_75/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_75/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_75/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_75/Tensordot/free:output:0<multi_head_self_attention_8/dense_75/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_75/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_75/Tensordot/concat?
4multi_head_self_attention_8/dense_75/Tensordot/stackPack<multi_head_self_attention_8/dense_75/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_75/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_75/Tensordot/stack?
8multi_head_self_attention_8/dense_75/Tensordot/transpose	Transpose.multi_head_self_attention_8/Reshape_3:output:0>multi_head_self_attention_8/dense_75/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2:
8multi_head_self_attention_8/dense_75/Tensordot/transpose?
6multi_head_self_attention_8/dense_75/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_75/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_75/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_75/Tensordot/Reshape?
5multi_head_self_attention_8/dense_75/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_75/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_75/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_75/Tensordot/MatMul?
6multi_head_self_attention_8/dense_75/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_75/Tensordot/Const_2?
<multi_head_self_attention_8/dense_75/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_75/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_75/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_75/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_75/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_75/Tensordot/concat_1?
.multi_head_self_attention_8/dense_75/TensordotReshape?multi_head_self_attention_8/dense_75/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_75/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_8/dense_75/Tensordot?
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_75_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_75/BiasAddBiasAdd7multi_head_self_attention_8/dense_75/Tensordot:output:0Cmulti_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2.
,multi_head_self_attention_8/dense_75/BiasAdd?
dropout_16/IdentityIdentity5multi_head_self_attention_8/dense_75/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_16/Identityo
addAddV2inputsdropout_16/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add?
5layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_16/moments/mean/reduction_indices?
#layer_normalization_16/moments/meanMeanadd:z:0>layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_16/moments/mean?
+layer_normalization_16/moments/StopGradientStopGradient,layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_16/moments/StopGradient?
0layer_normalization_16/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_16/moments/SquaredDifference?
9layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_16/moments/variance/reduction_indices?
'layer_normalization_16/moments/varianceMean4layer_normalization_16/moments/SquaredDifference:z:0Blayer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_16/moments/variance?
&layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_16/batchnorm/add/y?
$layer_normalization_16/batchnorm/addAddV20layer_normalization_16/moments/variance:output:0/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_16/batchnorm/add?
&layer_normalization_16/batchnorm/RsqrtRsqrt(layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_16/batchnorm/Rsqrt?
3layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_16/batchnorm/mul/ReadVariableOp?
$layer_normalization_16/batchnorm/mulMul*layer_normalization_16/batchnorm/Rsqrt:y:0;layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_16/batchnorm/mul?
&layer_normalization_16/batchnorm/mul_1Muladd:z:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/mul_1?
&layer_normalization_16/batchnorm/mul_2Mul,layer_normalization_16/moments/mean:output:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/mul_2?
/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_16/batchnorm/ReadVariableOp?
$layer_normalization_16/batchnorm/subSub7layer_normalization_16/batchnorm/ReadVariableOp:value:0*layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_16/batchnorm/sub?
&layer_normalization_16/batchnorm/add_1AddV2*layer_normalization_16/batchnorm/mul_1:z:0(layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/add_1?
.sequential_8/dense_76/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_76_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_8/dense_76/Tensordot/ReadVariableOp?
$sequential_8/dense_76/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_8/dense_76/Tensordot/axes?
$sequential_8/dense_76/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_8/dense_76/Tensordot/free?
%sequential_8/dense_76/Tensordot/ShapeShape*layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_8/dense_76/Tensordot/Shape?
-sequential_8/dense_76/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_76/Tensordot/GatherV2/axis?
(sequential_8/dense_76/Tensordot/GatherV2GatherV2.sequential_8/dense_76/Tensordot/Shape:output:0-sequential_8/dense_76/Tensordot/free:output:06sequential_8/dense_76/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_8/dense_76/Tensordot/GatherV2?
/sequential_8/dense_76/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_8/dense_76/Tensordot/GatherV2_1/axis?
*sequential_8/dense_76/Tensordot/GatherV2_1GatherV2.sequential_8/dense_76/Tensordot/Shape:output:0-sequential_8/dense_76/Tensordot/axes:output:08sequential_8/dense_76/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_8/dense_76/Tensordot/GatherV2_1?
%sequential_8/dense_76/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_8/dense_76/Tensordot/Const?
$sequential_8/dense_76/Tensordot/ProdProd1sequential_8/dense_76/Tensordot/GatherV2:output:0.sequential_8/dense_76/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_8/dense_76/Tensordot/Prod?
'sequential_8/dense_76/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_76/Tensordot/Const_1?
&sequential_8/dense_76/Tensordot/Prod_1Prod3sequential_8/dense_76/Tensordot/GatherV2_1:output:00sequential_8/dense_76/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_8/dense_76/Tensordot/Prod_1?
+sequential_8/dense_76/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_8/dense_76/Tensordot/concat/axis?
&sequential_8/dense_76/Tensordot/concatConcatV2-sequential_8/dense_76/Tensordot/free:output:0-sequential_8/dense_76/Tensordot/axes:output:04sequential_8/dense_76/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_76/Tensordot/concat?
%sequential_8/dense_76/Tensordot/stackPack-sequential_8/dense_76/Tensordot/Prod:output:0/sequential_8/dense_76/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/dense_76/Tensordot/stack?
)sequential_8/dense_76/Tensordot/transpose	Transpose*layer_normalization_16/batchnorm/add_1:z:0/sequential_8/dense_76/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_8/dense_76/Tensordot/transpose?
'sequential_8/dense_76/Tensordot/ReshapeReshape-sequential_8/dense_76/Tensordot/transpose:y:0.sequential_8/dense_76/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_8/dense_76/Tensordot/Reshape?
&sequential_8/dense_76/Tensordot/MatMulMatMul0sequential_8/dense_76/Tensordot/Reshape:output:06sequential_8/dense_76/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_8/dense_76/Tensordot/MatMul?
'sequential_8/dense_76/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_76/Tensordot/Const_2?
-sequential_8/dense_76/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_76/Tensordot/concat_1/axis?
(sequential_8/dense_76/Tensordot/concat_1ConcatV21sequential_8/dense_76/Tensordot/GatherV2:output:00sequential_8/dense_76/Tensordot/Const_2:output:06sequential_8/dense_76/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_8/dense_76/Tensordot/concat_1?
sequential_8/dense_76/TensordotReshape0sequential_8/dense_76/Tensordot/MatMul:product:01sequential_8/dense_76/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_8/dense_76/Tensordot?
,sequential_8/dense_76/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_76_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_8/dense_76/BiasAdd/ReadVariableOp?
sequential_8/dense_76/BiasAddBiasAdd(sequential_8/dense_76/Tensordot:output:04sequential_8/dense_76/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_76/BiasAdd?
sequential_8/dense_76/ReluRelu&sequential_8/dense_76/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_76/Relu?
.sequential_8/dense_77/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_77_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_8/dense_77/Tensordot/ReadVariableOp?
$sequential_8/dense_77/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_8/dense_77/Tensordot/axes?
$sequential_8/dense_77/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_8/dense_77/Tensordot/free?
%sequential_8/dense_77/Tensordot/ShapeShape(sequential_8/dense_76/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_8/dense_77/Tensordot/Shape?
-sequential_8/dense_77/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_77/Tensordot/GatherV2/axis?
(sequential_8/dense_77/Tensordot/GatherV2GatherV2.sequential_8/dense_77/Tensordot/Shape:output:0-sequential_8/dense_77/Tensordot/free:output:06sequential_8/dense_77/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_8/dense_77/Tensordot/GatherV2?
/sequential_8/dense_77/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_8/dense_77/Tensordot/GatherV2_1/axis?
*sequential_8/dense_77/Tensordot/GatherV2_1GatherV2.sequential_8/dense_77/Tensordot/Shape:output:0-sequential_8/dense_77/Tensordot/axes:output:08sequential_8/dense_77/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_8/dense_77/Tensordot/GatherV2_1?
%sequential_8/dense_77/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_8/dense_77/Tensordot/Const?
$sequential_8/dense_77/Tensordot/ProdProd1sequential_8/dense_77/Tensordot/GatherV2:output:0.sequential_8/dense_77/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_8/dense_77/Tensordot/Prod?
'sequential_8/dense_77/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_77/Tensordot/Const_1?
&sequential_8/dense_77/Tensordot/Prod_1Prod3sequential_8/dense_77/Tensordot/GatherV2_1:output:00sequential_8/dense_77/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_8/dense_77/Tensordot/Prod_1?
+sequential_8/dense_77/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_8/dense_77/Tensordot/concat/axis?
&sequential_8/dense_77/Tensordot/concatConcatV2-sequential_8/dense_77/Tensordot/free:output:0-sequential_8/dense_77/Tensordot/axes:output:04sequential_8/dense_77/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_77/Tensordot/concat?
%sequential_8/dense_77/Tensordot/stackPack-sequential_8/dense_77/Tensordot/Prod:output:0/sequential_8/dense_77/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/dense_77/Tensordot/stack?
)sequential_8/dense_77/Tensordot/transpose	Transpose(sequential_8/dense_76/Relu:activations:0/sequential_8/dense_77/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_8/dense_77/Tensordot/transpose?
'sequential_8/dense_77/Tensordot/ReshapeReshape-sequential_8/dense_77/Tensordot/transpose:y:0.sequential_8/dense_77/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_8/dense_77/Tensordot/Reshape?
&sequential_8/dense_77/Tensordot/MatMulMatMul0sequential_8/dense_77/Tensordot/Reshape:output:06sequential_8/dense_77/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_8/dense_77/Tensordot/MatMul?
'sequential_8/dense_77/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_77/Tensordot/Const_2?
-sequential_8/dense_77/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_77/Tensordot/concat_1/axis?
(sequential_8/dense_77/Tensordot/concat_1ConcatV21sequential_8/dense_77/Tensordot/GatherV2:output:00sequential_8/dense_77/Tensordot/Const_2:output:06sequential_8/dense_77/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_8/dense_77/Tensordot/concat_1?
sequential_8/dense_77/TensordotReshape0sequential_8/dense_77/Tensordot/MatMul:product:01sequential_8/dense_77/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_8/dense_77/Tensordot?
,sequential_8/dense_77/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_8/dense_77/BiasAdd/ReadVariableOp?
sequential_8/dense_77/BiasAddBiasAdd(sequential_8/dense_77/Tensordot:output:04sequential_8/dense_77/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_77/BiasAdd?
dropout_17/IdentityIdentity&sequential_8/dense_77/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_17/Identity?
add_1AddV2*layer_normalization_16/batchnorm/add_1:z:0dropout_17/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add_1?
5layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_17/moments/mean/reduction_indices?
#layer_normalization_17/moments/meanMean	add_1:z:0>layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_17/moments/mean?
+layer_normalization_17/moments/StopGradientStopGradient,layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_17/moments/StopGradient?
0layer_normalization_17/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_17/moments/SquaredDifference?
9layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_17/moments/variance/reduction_indices?
'layer_normalization_17/moments/varianceMean4layer_normalization_17/moments/SquaredDifference:z:0Blayer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_17/moments/variance?
&layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_17/batchnorm/add/y?
$layer_normalization_17/batchnorm/addAddV20layer_normalization_17/moments/variance:output:0/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_17/batchnorm/add?
&layer_normalization_17/batchnorm/RsqrtRsqrt(layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_17/batchnorm/Rsqrt?
3layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_17/batchnorm/mul/ReadVariableOp?
$layer_normalization_17/batchnorm/mulMul*layer_normalization_17/batchnorm/Rsqrt:y:0;layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_17/batchnorm/mul?
&layer_normalization_17/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/mul_1?
&layer_normalization_17/batchnorm/mul_2Mul,layer_normalization_17/moments/mean:output:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/mul_2?
/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_17/batchnorm/ReadVariableOp?
$layer_normalization_17/batchnorm/subSub7layer_normalization_17/batchnorm/ReadVariableOp:value:0*layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_17/batchnorm/sub?
&layer_normalization_17/batchnorm/add_1AddV2*layer_normalization_17/batchnorm/mul_1:z:0(layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/add_1?
IdentityIdentity*layer_normalization_17/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp0^layer_normalization_16/batchnorm/ReadVariableOp4^layer_normalization_16/batchnorm/mul/ReadVariableOp0^layer_normalization_17/batchnorm/ReadVariableOp4^layer_normalization_17/batchnorm/mul/ReadVariableOp<^multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp-^sequential_8/dense_76/BiasAdd/ReadVariableOp/^sequential_8/dense_76/Tensordot/ReadVariableOp-^sequential_8/dense_77/BiasAdd/ReadVariableOp/^sequential_8/dense_77/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2b
/layer_normalization_16/batchnorm/ReadVariableOp/layer_normalization_16/batchnorm/ReadVariableOp2j
3layer_normalization_16/batchnorm/mul/ReadVariableOp3layer_normalization_16/batchnorm/mul/ReadVariableOp2b
/layer_normalization_17/batchnorm/ReadVariableOp/layer_normalization_17/batchnorm/ReadVariableOp2j
3layer_normalization_17/batchnorm/mul/ReadVariableOp3layer_normalization_17/batchnorm/mul/ReadVariableOp2z
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp2\
,sequential_8/dense_76/BiasAdd/ReadVariableOp,sequential_8/dense_76/BiasAdd/ReadVariableOp2`
.sequential_8/dense_76/Tensordot/ReadVariableOp.sequential_8/dense_76/Tensordot/ReadVariableOp2\
,sequential_8/dense_77/BiasAdd/ReadVariableOp,sequential_8/dense_77/BiasAdd/ReadVariableOp2`
.sequential_8/dense_77/Tensordot/ReadVariableOp.sequential_8/dense_77/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
?
*__inference_dense_80_layer_call_fn_4834568

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
E__inference_dense_80_layer_call_and_return_conditional_losses_48321352
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
?
?
.__inference_sequential_8_layer_call_fn_4831688
dense_76_input
unknown:  
	unknown_0: 
	unknown_1:  
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_76_inputunknown	unknown_0	unknown_1	unknown_2*
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_48316642
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
_user_specified_namedense_76_input
?!
?
E__inference_dense_76_layer_call_and_return_conditional_losses_4831561

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
?
?
,__inference_aux_output_layer_call_fn_4834495

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
G__inference_aux_output_layer_call_and_return_conditional_losses_48320752
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
?<
?
D__inference_model_8_layer_call_and_return_conditional_losses_4833031
input_9
	aux_input8
&token_and_position_embedding_8_4832964:( 8
&token_and_position_embedding_8_4832966: -
transformer_block_8_4832969:  )
transformer_block_8_4832971: -
transformer_block_8_4832973:  )
transformer_block_8_4832975: -
transformer_block_8_4832977:  )
transformer_block_8_4832979: -
transformer_block_8_4832981:  )
transformer_block_8_4832983: )
transformer_block_8_4832985: )
transformer_block_8_4832987: -
transformer_block_8_4832989:  )
transformer_block_8_4832991: -
transformer_block_8_4832993:  )
transformer_block_8_4832995: )
transformer_block_8_4832997: )
transformer_block_8_4832999: $
aux_output_4833003:  
aux_output_4833005:"
dense_78_4833009:@
dense_78_4833011:@"
dense_79_4833014:@@
dense_79_4833016:@"
dense_80_4833019:@@
dense_80_4833021:@%
main_output_4833024:@!
main_output_4833026:
identity

identity_1??"aux_output/StatefulPartitionedCall? dense_78/StatefulPartitionedCall? dense_79/StatefulPartitionedCall? dense_80/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?6token_and_position_embedding_8/StatefulPartitionedCall?+transformer_block_8/StatefulPartitionedCall?
6token_and_position_embedding_8/StatefulPartitionedCallStatefulPartitionedCallinput_9&token_and_position_embedding_8_4832964&token_and_position_embedding_8_4832966*
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
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_483177328
6token_and_position_embedding_8/StatefulPartitionedCall?
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_8/StatefulPartitionedCall:output:0transformer_block_8_4832969transformer_block_8_4832971transformer_block_8_4832973transformer_block_8_4832975transformer_block_8_4832977transformer_block_8_4832979transformer_block_8_4832981transformer_block_8_4832983transformer_block_8_4832985transformer_block_8_4832987transformer_block_8_4832989transformer_block_8_4832991transformer_block_8_4832993transformer_block_8_4832995transformer_block_8_4832997transformer_block_8_4832999*
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
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_48325812-
+transformer_block_8/StatefulPartitionedCall?
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
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
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_48320622,
*global_average_pooling1d_8/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0aux_output_4833003aux_output_4833005*
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
G__inference_aux_output_layer_call_and_return_conditional_losses_48320752$
"aux_output/StatefulPartitionedCall?
concatenate_8/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0	aux_input*
Tin
2*
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
GPU 2J 8? *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_48320882
concatenate_8/PartitionedCall?
 dense_78/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_78_4833009dense_78_4833011*
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
E__inference_dense_78_layer_call_and_return_conditional_losses_48321012"
 dense_78/StatefulPartitionedCall?
 dense_79/StatefulPartitionedCallStatefulPartitionedCall)dense_78/StatefulPartitionedCall:output:0dense_79_4833014dense_79_4833016*
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
E__inference_dense_79_layer_call_and_return_conditional_losses_48321182"
 dense_79/StatefulPartitionedCall?
 dense_80/StatefulPartitionedCallStatefulPartitionedCall)dense_79/StatefulPartitionedCall:output:0dense_80_4833019dense_80_4833021*
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
E__inference_dense_80_layer_call_and_return_conditional_losses_48321352"
 dense_80/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0main_output_4833024main_output_4833026*
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
H__inference_main_output_layer_call_and_return_conditional_losses_48321522%
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
NoOpNoOp#^aux_output/StatefulPartitionedCall!^dense_78/StatefulPartitionedCall!^dense_79/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall$^main_output/StatefulPartitionedCall7^token_and_position_embedding_8/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2p
6token_and_position_embedding_8/StatefulPartitionedCall6token_and_position_embedding_8/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:P L
'
_output_shapes
:?????????(
!
_user_specified_name	input_9:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input
??
?
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_4834464

inputsX
Fmulti_head_self_attention_8_dense_72_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_72_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_73_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_73_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_74_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_74_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_75_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_75_biasadd_readvariableop_resource: J
<layer_normalization_16_batchnorm_mul_readvariableop_resource: F
8layer_normalization_16_batchnorm_readvariableop_resource: I
7sequential_8_dense_76_tensordot_readvariableop_resource:  C
5sequential_8_dense_76_biasadd_readvariableop_resource: I
7sequential_8_dense_77_tensordot_readvariableop_resource:  C
5sequential_8_dense_77_biasadd_readvariableop_resource: J
<layer_normalization_17_batchnorm_mul_readvariableop_resource: F
8layer_normalization_17_batchnorm_readvariableop_resource: 
identity??/layer_normalization_16/batchnorm/ReadVariableOp?3layer_normalization_16/batchnorm/mul/ReadVariableOp?/layer_normalization_17/batchnorm/ReadVariableOp?3layer_normalization_17/batchnorm/mul/ReadVariableOp?;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?,sequential_8/dense_76/BiasAdd/ReadVariableOp?.sequential_8/dense_76/Tensordot/ReadVariableOp?,sequential_8/dense_77/BiasAdd/ReadVariableOp?.sequential_8/dense_77/Tensordot/ReadVariableOp|
!multi_head_self_attention_8/ShapeShapeinputs*
T0*
_output_shapes
:2#
!multi_head_self_attention_8/Shape?
/multi_head_self_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention_8/strided_slice/stack?
1multi_head_self_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_8/strided_slice/stack_1?
1multi_head_self_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_8/strided_slice/stack_2?
)multi_head_self_attention_8/strided_sliceStridedSlice*multi_head_self_attention_8/Shape:output:08multi_head_self_attention_8/strided_slice/stack:output:0:multi_head_self_attention_8/strided_slice/stack_1:output:0:multi_head_self_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention_8/strided_slice?
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_72_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_72/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_72/Tensordot/axes?
3multi_head_self_attention_8/dense_72/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_72/Tensordot/free?
4multi_head_self_attention_8/dense_72/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_72/Tensordot/Shape?
<multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_72/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_72/Tensordot/free:output:0Emulti_head_self_attention_8/dense_72/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_72/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_72/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_72/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_72/Tensordot/Const?
3multi_head_self_attention_8/dense_72/Tensordot/ProdProd@multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_72/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_72/Tensordot/Prod?
6multi_head_self_attention_8/dense_72/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_72/Tensordot/Const_1?
5multi_head_self_attention_8/dense_72/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_72/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_72/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_72/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_72/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_72/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_72/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_72/Tensordot/free:output:0<multi_head_self_attention_8/dense_72/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_72/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_72/Tensordot/concat?
4multi_head_self_attention_8/dense_72/Tensordot/stackPack<multi_head_self_attention_8/dense_72/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_72/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_72/Tensordot/stack?
8multi_head_self_attention_8/dense_72/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_72/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_72/Tensordot/transpose?
6multi_head_self_attention_8/dense_72/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_72/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_72/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_72/Tensordot/Reshape?
5multi_head_self_attention_8/dense_72/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_72/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_72/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_72/Tensordot/MatMul?
6multi_head_self_attention_8/dense_72/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_72/Tensordot/Const_2?
<multi_head_self_attention_8/dense_72/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_72/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_72/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_72/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_72/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_72/Tensordot/concat_1?
.multi_head_self_attention_8/dense_72/TensordotReshape?multi_head_self_attention_8/dense_72/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_72/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_72/Tensordot?
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_72_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_72/BiasAddBiasAdd7multi_head_self_attention_8/dense_72/Tensordot:output:0Cmulti_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_72/BiasAdd?
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_73_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_73/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_73/Tensordot/axes?
3multi_head_self_attention_8/dense_73/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_73/Tensordot/free?
4multi_head_self_attention_8/dense_73/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_73/Tensordot/Shape?
<multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_73/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_73/Tensordot/free:output:0Emulti_head_self_attention_8/dense_73/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_73/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_73/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_73/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_73/Tensordot/Const?
3multi_head_self_attention_8/dense_73/Tensordot/ProdProd@multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_73/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_73/Tensordot/Prod?
6multi_head_self_attention_8/dense_73/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_73/Tensordot/Const_1?
5multi_head_self_attention_8/dense_73/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_73/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_73/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_73/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_73/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_73/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_73/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_73/Tensordot/free:output:0<multi_head_self_attention_8/dense_73/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_73/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_73/Tensordot/concat?
4multi_head_self_attention_8/dense_73/Tensordot/stackPack<multi_head_self_attention_8/dense_73/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_73/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_73/Tensordot/stack?
8multi_head_self_attention_8/dense_73/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_73/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_73/Tensordot/transpose?
6multi_head_self_attention_8/dense_73/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_73/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_73/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_73/Tensordot/Reshape?
5multi_head_self_attention_8/dense_73/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_73/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_73/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_73/Tensordot/MatMul?
6multi_head_self_attention_8/dense_73/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_73/Tensordot/Const_2?
<multi_head_self_attention_8/dense_73/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_73/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_73/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_73/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_73/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_73/Tensordot/concat_1?
.multi_head_self_attention_8/dense_73/TensordotReshape?multi_head_self_attention_8/dense_73/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_73/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_73/Tensordot?
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_73/BiasAddBiasAdd7multi_head_self_attention_8/dense_73/Tensordot:output:0Cmulti_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_73/BiasAdd?
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_74_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_74/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_74/Tensordot/axes?
3multi_head_self_attention_8/dense_74/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_74/Tensordot/free?
4multi_head_self_attention_8/dense_74/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_74/Tensordot/Shape?
<multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_74/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_74/Tensordot/free:output:0Emulti_head_self_attention_8/dense_74/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_74/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_74/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_74/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_74/Tensordot/Const?
3multi_head_self_attention_8/dense_74/Tensordot/ProdProd@multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_74/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_74/Tensordot/Prod?
6multi_head_self_attention_8/dense_74/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_74/Tensordot/Const_1?
5multi_head_self_attention_8/dense_74/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_74/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_74/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_74/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_74/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_74/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_74/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_74/Tensordot/free:output:0<multi_head_self_attention_8/dense_74/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_74/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_74/Tensordot/concat?
4multi_head_self_attention_8/dense_74/Tensordot/stackPack<multi_head_self_attention_8/dense_74/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_74/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_74/Tensordot/stack?
8multi_head_self_attention_8/dense_74/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_74/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_74/Tensordot/transpose?
6multi_head_self_attention_8/dense_74/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_74/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_74/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_74/Tensordot/Reshape?
5multi_head_self_attention_8/dense_74/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_74/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_74/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_74/Tensordot/MatMul?
6multi_head_self_attention_8/dense_74/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_74/Tensordot/Const_2?
<multi_head_self_attention_8/dense_74/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_74/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_74/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_74/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_74/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_74/Tensordot/concat_1?
.multi_head_self_attention_8/dense_74/TensordotReshape?multi_head_self_attention_8/dense_74/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_74/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_74/Tensordot?
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_74/BiasAddBiasAdd7multi_head_self_attention_8/dense_74/Tensordot:output:0Cmulti_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_74/BiasAdd?
+multi_head_self_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+multi_head_self_attention_8/Reshape/shape/1?
+multi_head_self_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_8/Reshape/shape/2?
+multi_head_self_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_8/Reshape/shape/3?
)multi_head_self_attention_8/Reshape/shapePack2multi_head_self_attention_8/strided_slice:output:04multi_head_self_attention_8/Reshape/shape/1:output:04multi_head_self_attention_8/Reshape/shape/2:output:04multi_head_self_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention_8/Reshape/shape?
#multi_head_self_attention_8/ReshapeReshape5multi_head_self_attention_8/dense_72/BiasAdd:output:02multi_head_self_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2%
#multi_head_self_attention_8/Reshape?
*multi_head_self_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention_8/transpose/perm?
%multi_head_self_attention_8/transpose	Transpose,multi_head_self_attention_8/Reshape:output:03multi_head_self_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/transpose?
-multi_head_self_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_1/shape/1?
-multi_head_self_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_1/shape/2?
-multi_head_self_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_1/shape/3?
+multi_head_self_attention_8/Reshape_1/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_1/shape/1:output:06multi_head_self_attention_8/Reshape_1/shape/2:output:06multi_head_self_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_1/shape?
%multi_head_self_attention_8/Reshape_1Reshape5multi_head_self_attention_8/dense_73/BiasAdd:output:04multi_head_self_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/Reshape_1?
,multi_head_self_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_1/perm?
'multi_head_self_attention_8/transpose_1	Transpose.multi_head_self_attention_8/Reshape_1:output:05multi_head_self_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_1?
-multi_head_self_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_2/shape/1?
-multi_head_self_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_2/shape/2?
-multi_head_self_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_2/shape/3?
+multi_head_self_attention_8/Reshape_2/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_2/shape/1:output:06multi_head_self_attention_8/Reshape_2/shape/2:output:06multi_head_self_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_2/shape?
%multi_head_self_attention_8/Reshape_2Reshape5multi_head_self_attention_8/dense_74/BiasAdd:output:04multi_head_self_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/Reshape_2?
,multi_head_self_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_2/perm?
'multi_head_self_attention_8/transpose_2	Transpose.multi_head_self_attention_8/Reshape_2:output:05multi_head_self_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_2?
"multi_head_self_attention_8/MatMulBatchMatMulV2)multi_head_self_attention_8/transpose:y:0+multi_head_self_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2$
"multi_head_self_attention_8/MatMul?
#multi_head_self_attention_8/Shape_1Shape+multi_head_self_attention_8/transpose_1:y:0*
T0*
_output_shapes
:2%
#multi_head_self_attention_8/Shape_1?
1multi_head_self_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1multi_head_self_attention_8/strided_slice_1/stack?
3multi_head_self_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention_8/strided_slice_1/stack_1?
3multi_head_self_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/strided_slice_1/stack_2?
+multi_head_self_attention_8/strided_slice_1StridedSlice,multi_head_self_attention_8/Shape_1:output:0:multi_head_self_attention_8/strided_slice_1/stack:output:0<multi_head_self_attention_8/strided_slice_1/stack_1:output:0<multi_head_self_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+multi_head_self_attention_8/strided_slice_1?
 multi_head_self_attention_8/CastCast4multi_head_self_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 multi_head_self_attention_8/Cast?
 multi_head_self_attention_8/SqrtSqrt$multi_head_self_attention_8/Cast:y:0*
T0*
_output_shapes
: 2"
 multi_head_self_attention_8/Sqrt?
#multi_head_self_attention_8/truedivRealDiv+multi_head_self_attention_8/MatMul:output:0$multi_head_self_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_8/truediv?
#multi_head_self_attention_8/SoftmaxSoftmax'multi_head_self_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_8/Softmax?
$multi_head_self_attention_8/MatMul_1BatchMatMulV2-multi_head_self_attention_8/Softmax:softmax:0+multi_head_self_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_8/MatMul_1?
,multi_head_self_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_3/perm?
'multi_head_self_attention_8/transpose_3	Transpose-multi_head_self_attention_8/MatMul_1:output:05multi_head_self_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_3?
-multi_head_self_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_3/shape/1?
-multi_head_self_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2/
-multi_head_self_attention_8/Reshape_3/shape/2?
+multi_head_self_attention_8/Reshape_3/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_3/shape/1:output:06multi_head_self_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_3/shape?
%multi_head_self_attention_8/Reshape_3Reshape+multi_head_self_attention_8/transpose_3:y:04multi_head_self_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2'
%multi_head_self_attention_8/Reshape_3?
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_75_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_75/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_75/Tensordot/axes?
3multi_head_self_attention_8/dense_75/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_75/Tensordot/free?
4multi_head_self_attention_8/dense_75/Tensordot/ShapeShape.multi_head_self_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_75/Tensordot/Shape?
<multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_75/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_75/Tensordot/free:output:0Emulti_head_self_attention_8/dense_75/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_75/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_75/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_75/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_75/Tensordot/Const?
3multi_head_self_attention_8/dense_75/Tensordot/ProdProd@multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_75/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_75/Tensordot/Prod?
6multi_head_self_attention_8/dense_75/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_75/Tensordot/Const_1?
5multi_head_self_attention_8/dense_75/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_75/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_75/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_75/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_75/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_75/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_75/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_75/Tensordot/free:output:0<multi_head_self_attention_8/dense_75/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_75/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_75/Tensordot/concat?
4multi_head_self_attention_8/dense_75/Tensordot/stackPack<multi_head_self_attention_8/dense_75/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_75/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_75/Tensordot/stack?
8multi_head_self_attention_8/dense_75/Tensordot/transpose	Transpose.multi_head_self_attention_8/Reshape_3:output:0>multi_head_self_attention_8/dense_75/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2:
8multi_head_self_attention_8/dense_75/Tensordot/transpose?
6multi_head_self_attention_8/dense_75/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_75/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_75/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_75/Tensordot/Reshape?
5multi_head_self_attention_8/dense_75/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_75/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_75/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_75/Tensordot/MatMul?
6multi_head_self_attention_8/dense_75/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_75/Tensordot/Const_2?
<multi_head_self_attention_8/dense_75/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_75/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_75/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_75/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_75/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_75/Tensordot/concat_1?
.multi_head_self_attention_8/dense_75/TensordotReshape?multi_head_self_attention_8/dense_75/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_75/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_8/dense_75/Tensordot?
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_75_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_75/BiasAddBiasAdd7multi_head_self_attention_8/dense_75/Tensordot:output:0Cmulti_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2.
,multi_head_self_attention_8/dense_75/BiasAddy
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_16/dropout/Const?
dropout_16/dropout/MulMul5multi_head_self_attention_8/dense_75/BiasAdd:output:0!dropout_16/dropout/Const:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_16/dropout/Mul?
dropout_16/dropout/ShapeShape5multi_head_self_attention_8/dense_75/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*4
_output_shapes"
 :?????????????????? *
dtype021
/dropout_16/dropout/random_uniform/RandomUniform?
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_16/dropout/GreaterEqual/y?
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :?????????????????? 2!
dropout_16/dropout/GreaterEqual?
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :?????????????????? 2
dropout_16/dropout/Cast?
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_16/dropout/Mul_1o
addAddV2inputsdropout_16/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add?
5layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_16/moments/mean/reduction_indices?
#layer_normalization_16/moments/meanMeanadd:z:0>layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_16/moments/mean?
+layer_normalization_16/moments/StopGradientStopGradient,layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_16/moments/StopGradient?
0layer_normalization_16/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_16/moments/SquaredDifference?
9layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_16/moments/variance/reduction_indices?
'layer_normalization_16/moments/varianceMean4layer_normalization_16/moments/SquaredDifference:z:0Blayer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_16/moments/variance?
&layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_16/batchnorm/add/y?
$layer_normalization_16/batchnorm/addAddV20layer_normalization_16/moments/variance:output:0/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_16/batchnorm/add?
&layer_normalization_16/batchnorm/RsqrtRsqrt(layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_16/batchnorm/Rsqrt?
3layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_16/batchnorm/mul/ReadVariableOp?
$layer_normalization_16/batchnorm/mulMul*layer_normalization_16/batchnorm/Rsqrt:y:0;layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_16/batchnorm/mul?
&layer_normalization_16/batchnorm/mul_1Muladd:z:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/mul_1?
&layer_normalization_16/batchnorm/mul_2Mul,layer_normalization_16/moments/mean:output:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/mul_2?
/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_16/batchnorm/ReadVariableOp?
$layer_normalization_16/batchnorm/subSub7layer_normalization_16/batchnorm/ReadVariableOp:value:0*layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_16/batchnorm/sub?
&layer_normalization_16/batchnorm/add_1AddV2*layer_normalization_16/batchnorm/mul_1:z:0(layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/add_1?
.sequential_8/dense_76/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_76_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_8/dense_76/Tensordot/ReadVariableOp?
$sequential_8/dense_76/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_8/dense_76/Tensordot/axes?
$sequential_8/dense_76/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_8/dense_76/Tensordot/free?
%sequential_8/dense_76/Tensordot/ShapeShape*layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_8/dense_76/Tensordot/Shape?
-sequential_8/dense_76/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_76/Tensordot/GatherV2/axis?
(sequential_8/dense_76/Tensordot/GatherV2GatherV2.sequential_8/dense_76/Tensordot/Shape:output:0-sequential_8/dense_76/Tensordot/free:output:06sequential_8/dense_76/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_8/dense_76/Tensordot/GatherV2?
/sequential_8/dense_76/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_8/dense_76/Tensordot/GatherV2_1/axis?
*sequential_8/dense_76/Tensordot/GatherV2_1GatherV2.sequential_8/dense_76/Tensordot/Shape:output:0-sequential_8/dense_76/Tensordot/axes:output:08sequential_8/dense_76/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_8/dense_76/Tensordot/GatherV2_1?
%sequential_8/dense_76/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_8/dense_76/Tensordot/Const?
$sequential_8/dense_76/Tensordot/ProdProd1sequential_8/dense_76/Tensordot/GatherV2:output:0.sequential_8/dense_76/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_8/dense_76/Tensordot/Prod?
'sequential_8/dense_76/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_76/Tensordot/Const_1?
&sequential_8/dense_76/Tensordot/Prod_1Prod3sequential_8/dense_76/Tensordot/GatherV2_1:output:00sequential_8/dense_76/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_8/dense_76/Tensordot/Prod_1?
+sequential_8/dense_76/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_8/dense_76/Tensordot/concat/axis?
&sequential_8/dense_76/Tensordot/concatConcatV2-sequential_8/dense_76/Tensordot/free:output:0-sequential_8/dense_76/Tensordot/axes:output:04sequential_8/dense_76/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_76/Tensordot/concat?
%sequential_8/dense_76/Tensordot/stackPack-sequential_8/dense_76/Tensordot/Prod:output:0/sequential_8/dense_76/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/dense_76/Tensordot/stack?
)sequential_8/dense_76/Tensordot/transpose	Transpose*layer_normalization_16/batchnorm/add_1:z:0/sequential_8/dense_76/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_8/dense_76/Tensordot/transpose?
'sequential_8/dense_76/Tensordot/ReshapeReshape-sequential_8/dense_76/Tensordot/transpose:y:0.sequential_8/dense_76/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_8/dense_76/Tensordot/Reshape?
&sequential_8/dense_76/Tensordot/MatMulMatMul0sequential_8/dense_76/Tensordot/Reshape:output:06sequential_8/dense_76/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_8/dense_76/Tensordot/MatMul?
'sequential_8/dense_76/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_76/Tensordot/Const_2?
-sequential_8/dense_76/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_76/Tensordot/concat_1/axis?
(sequential_8/dense_76/Tensordot/concat_1ConcatV21sequential_8/dense_76/Tensordot/GatherV2:output:00sequential_8/dense_76/Tensordot/Const_2:output:06sequential_8/dense_76/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_8/dense_76/Tensordot/concat_1?
sequential_8/dense_76/TensordotReshape0sequential_8/dense_76/Tensordot/MatMul:product:01sequential_8/dense_76/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_8/dense_76/Tensordot?
,sequential_8/dense_76/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_76_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_8/dense_76/BiasAdd/ReadVariableOp?
sequential_8/dense_76/BiasAddBiasAdd(sequential_8/dense_76/Tensordot:output:04sequential_8/dense_76/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_76/BiasAdd?
sequential_8/dense_76/ReluRelu&sequential_8/dense_76/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_76/Relu?
.sequential_8/dense_77/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_77_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_8/dense_77/Tensordot/ReadVariableOp?
$sequential_8/dense_77/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_8/dense_77/Tensordot/axes?
$sequential_8/dense_77/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_8/dense_77/Tensordot/free?
%sequential_8/dense_77/Tensordot/ShapeShape(sequential_8/dense_76/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_8/dense_77/Tensordot/Shape?
-sequential_8/dense_77/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_77/Tensordot/GatherV2/axis?
(sequential_8/dense_77/Tensordot/GatherV2GatherV2.sequential_8/dense_77/Tensordot/Shape:output:0-sequential_8/dense_77/Tensordot/free:output:06sequential_8/dense_77/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_8/dense_77/Tensordot/GatherV2?
/sequential_8/dense_77/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_8/dense_77/Tensordot/GatherV2_1/axis?
*sequential_8/dense_77/Tensordot/GatherV2_1GatherV2.sequential_8/dense_77/Tensordot/Shape:output:0-sequential_8/dense_77/Tensordot/axes:output:08sequential_8/dense_77/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_8/dense_77/Tensordot/GatherV2_1?
%sequential_8/dense_77/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_8/dense_77/Tensordot/Const?
$sequential_8/dense_77/Tensordot/ProdProd1sequential_8/dense_77/Tensordot/GatherV2:output:0.sequential_8/dense_77/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_8/dense_77/Tensordot/Prod?
'sequential_8/dense_77/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_77/Tensordot/Const_1?
&sequential_8/dense_77/Tensordot/Prod_1Prod3sequential_8/dense_77/Tensordot/GatherV2_1:output:00sequential_8/dense_77/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_8/dense_77/Tensordot/Prod_1?
+sequential_8/dense_77/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_8/dense_77/Tensordot/concat/axis?
&sequential_8/dense_77/Tensordot/concatConcatV2-sequential_8/dense_77/Tensordot/free:output:0-sequential_8/dense_77/Tensordot/axes:output:04sequential_8/dense_77/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_77/Tensordot/concat?
%sequential_8/dense_77/Tensordot/stackPack-sequential_8/dense_77/Tensordot/Prod:output:0/sequential_8/dense_77/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/dense_77/Tensordot/stack?
)sequential_8/dense_77/Tensordot/transpose	Transpose(sequential_8/dense_76/Relu:activations:0/sequential_8/dense_77/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_8/dense_77/Tensordot/transpose?
'sequential_8/dense_77/Tensordot/ReshapeReshape-sequential_8/dense_77/Tensordot/transpose:y:0.sequential_8/dense_77/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_8/dense_77/Tensordot/Reshape?
&sequential_8/dense_77/Tensordot/MatMulMatMul0sequential_8/dense_77/Tensordot/Reshape:output:06sequential_8/dense_77/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_8/dense_77/Tensordot/MatMul?
'sequential_8/dense_77/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_77/Tensordot/Const_2?
-sequential_8/dense_77/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_77/Tensordot/concat_1/axis?
(sequential_8/dense_77/Tensordot/concat_1ConcatV21sequential_8/dense_77/Tensordot/GatherV2:output:00sequential_8/dense_77/Tensordot/Const_2:output:06sequential_8/dense_77/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_8/dense_77/Tensordot/concat_1?
sequential_8/dense_77/TensordotReshape0sequential_8/dense_77/Tensordot/MatMul:product:01sequential_8/dense_77/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_8/dense_77/Tensordot?
,sequential_8/dense_77/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_8/dense_77/BiasAdd/ReadVariableOp?
sequential_8/dense_77/BiasAddBiasAdd(sequential_8/dense_77/Tensordot:output:04sequential_8/dense_77/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_77/BiasAddy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_17/dropout/Const?
dropout_17/dropout/MulMul&sequential_8/dense_77/BiasAdd:output:0!dropout_17/dropout/Const:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_17/dropout/Mul?
dropout_17/dropout/ShapeShape&sequential_8/dense_77/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape?
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????( *
dtype021
/dropout_17/dropout/random_uniform/RandomUniform?
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_17/dropout/GreaterEqual/y?
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????( 2!
dropout_17/dropout/GreaterEqual?
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????( 2
dropout_17/dropout/Cast?
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????( 2
dropout_17/dropout/Mul_1?
add_1AddV2*layer_normalization_16/batchnorm/add_1:z:0dropout_17/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????( 2
add_1?
5layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_17/moments/mean/reduction_indices?
#layer_normalization_17/moments/meanMean	add_1:z:0>layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_17/moments/mean?
+layer_normalization_17/moments/StopGradientStopGradient,layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_17/moments/StopGradient?
0layer_normalization_17/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_17/moments/SquaredDifference?
9layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_17/moments/variance/reduction_indices?
'layer_normalization_17/moments/varianceMean4layer_normalization_17/moments/SquaredDifference:z:0Blayer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_17/moments/variance?
&layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_17/batchnorm/add/y?
$layer_normalization_17/batchnorm/addAddV20layer_normalization_17/moments/variance:output:0/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_17/batchnorm/add?
&layer_normalization_17/batchnorm/RsqrtRsqrt(layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_17/batchnorm/Rsqrt?
3layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_17/batchnorm/mul/ReadVariableOp?
$layer_normalization_17/batchnorm/mulMul*layer_normalization_17/batchnorm/Rsqrt:y:0;layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_17/batchnorm/mul?
&layer_normalization_17/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/mul_1?
&layer_normalization_17/batchnorm/mul_2Mul,layer_normalization_17/moments/mean:output:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/mul_2?
/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_17/batchnorm/ReadVariableOp?
$layer_normalization_17/batchnorm/subSub7layer_normalization_17/batchnorm/ReadVariableOp:value:0*layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_17/batchnorm/sub?
&layer_normalization_17/batchnorm/add_1AddV2*layer_normalization_17/batchnorm/mul_1:z:0(layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/add_1?
IdentityIdentity*layer_normalization_17/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp0^layer_normalization_16/batchnorm/ReadVariableOp4^layer_normalization_16/batchnorm/mul/ReadVariableOp0^layer_normalization_17/batchnorm/ReadVariableOp4^layer_normalization_17/batchnorm/mul/ReadVariableOp<^multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp-^sequential_8/dense_76/BiasAdd/ReadVariableOp/^sequential_8/dense_76/Tensordot/ReadVariableOp-^sequential_8/dense_77/BiasAdd/ReadVariableOp/^sequential_8/dense_77/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2b
/layer_normalization_16/batchnorm/ReadVariableOp/layer_normalization_16/batchnorm/ReadVariableOp2j
3layer_normalization_16/batchnorm/mul/ReadVariableOp3layer_normalization_16/batchnorm/mul/ReadVariableOp2b
/layer_normalization_17/batchnorm/ReadVariableOp/layer_normalization_17/batchnorm/ReadVariableOp2j
3layer_normalization_17/batchnorm/mul/ReadVariableOp3layer_normalization_17/batchnorm/mul/ReadVariableOp2z
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp2\
,sequential_8/dense_76/BiasAdd/ReadVariableOp,sequential_8/dense_76/BiasAdd/ReadVariableOp2`
.sequential_8/dense_76/Tensordot/ReadVariableOp.sequential_8/dense_76/Tensordot/ReadVariableOp2\
,sequential_8/dense_77/BiasAdd/ReadVariableOp,sequential_8/dense_77/BiasAdd/ReadVariableOp2`
.sequential_8/dense_77/Tensordot/ReadVariableOp.sequential_8/dense_77/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
[
/__inference_concatenate_8_layer_call_fn_4834512
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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_48320882
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
5__inference_transformer_block_8_layer_call_fn_4833962

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
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_48325812
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
X
<__inference_global_average_pooling1d_8_layer_call_fn_4834474

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
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_48320622
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
?
?
E__inference_dense_79_layer_call_and_return_conditional_losses_4834559

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
D__inference_model_8_layer_call_and_return_conditional_losses_4832764

inputs
inputs_18
&token_and_position_embedding_8_4832697:( 8
&token_and_position_embedding_8_4832699: -
transformer_block_8_4832702:  )
transformer_block_8_4832704: -
transformer_block_8_4832706:  )
transformer_block_8_4832708: -
transformer_block_8_4832710:  )
transformer_block_8_4832712: -
transformer_block_8_4832714:  )
transformer_block_8_4832716: )
transformer_block_8_4832718: )
transformer_block_8_4832720: -
transformer_block_8_4832722:  )
transformer_block_8_4832724: -
transformer_block_8_4832726:  )
transformer_block_8_4832728: )
transformer_block_8_4832730: )
transformer_block_8_4832732: $
aux_output_4832736:  
aux_output_4832738:"
dense_78_4832742:@
dense_78_4832744:@"
dense_79_4832747:@@
dense_79_4832749:@"
dense_80_4832752:@@
dense_80_4832754:@%
main_output_4832757:@!
main_output_4832759:
identity

identity_1??"aux_output/StatefulPartitionedCall? dense_78/StatefulPartitionedCall? dense_79/StatefulPartitionedCall? dense_80/StatefulPartitionedCall?#main_output/StatefulPartitionedCall?6token_and_position_embedding_8/StatefulPartitionedCall?+transformer_block_8/StatefulPartitionedCall?
6token_and_position_embedding_8/StatefulPartitionedCallStatefulPartitionedCallinputs&token_and_position_embedding_8_4832697&token_and_position_embedding_8_4832699*
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
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_483177328
6token_and_position_embedding_8/StatefulPartitionedCall?
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_8/StatefulPartitionedCall:output:0transformer_block_8_4832702transformer_block_8_4832704transformer_block_8_4832706transformer_block_8_4832708transformer_block_8_4832710transformer_block_8_4832712transformer_block_8_4832714transformer_block_8_4832716transformer_block_8_4832718transformer_block_8_4832720transformer_block_8_4832722transformer_block_8_4832724transformer_block_8_4832726transformer_block_8_4832728transformer_block_8_4832730transformer_block_8_4832732*
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
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_48325812-
+transformer_block_8/StatefulPartitionedCall?
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
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
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_48320622,
*global_average_pooling1d_8/PartitionedCall?
"aux_output/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0aux_output_4832736aux_output_4832738*
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
G__inference_aux_output_layer_call_and_return_conditional_losses_48320752$
"aux_output/StatefulPartitionedCall?
concatenate_8/PartitionedCallPartitionedCall+aux_output/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
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
GPU 2J 8? *S
fNRL
J__inference_concatenate_8_layer_call_and_return_conditional_losses_48320882
concatenate_8/PartitionedCall?
 dense_78/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0dense_78_4832742dense_78_4832744*
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
E__inference_dense_78_layer_call_and_return_conditional_losses_48321012"
 dense_78/StatefulPartitionedCall?
 dense_79/StatefulPartitionedCallStatefulPartitionedCall)dense_78/StatefulPartitionedCall:output:0dense_79_4832747dense_79_4832749*
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
E__inference_dense_79_layer_call_and_return_conditional_losses_48321182"
 dense_79/StatefulPartitionedCall?
 dense_80/StatefulPartitionedCallStatefulPartitionedCall)dense_79/StatefulPartitionedCall:output:0dense_80_4832752dense_80_4832754*
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
E__inference_dense_80_layer_call_and_return_conditional_losses_48321352"
 dense_80/StatefulPartitionedCall?
#main_output/StatefulPartitionedCallStatefulPartitionedCall)dense_80/StatefulPartitionedCall:output:0main_output_4832757main_output_4832759*
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
H__inference_main_output_layer_call_and_return_conditional_losses_48321522%
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
NoOpNoOp#^aux_output/StatefulPartitionedCall!^dense_78/StatefulPartitionedCall!^dense_79/StatefulPartitionedCall!^dense_80/StatefulPartitionedCall$^main_output/StatefulPartitionedCall7^token_and_position_embedding_8/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"aux_output/StatefulPartitionedCall"aux_output/StatefulPartitionedCall2D
 dense_78/StatefulPartitionedCall dense_78/StatefulPartitionedCall2D
 dense_79/StatefulPartitionedCall dense_79/StatefulPartitionedCall2D
 dense_80/StatefulPartitionedCall dense_80/StatefulPartitionedCall2J
#main_output/StatefulPartitionedCall#main_output/StatefulPartitionedCall2p
6token_and_position_embedding_8/StatefulPartitionedCall6token_and_position_embedding_8/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
s
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_4831726

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
?
?
I__inference_sequential_8_layer_call_and_return_conditional_losses_4831702
dense_76_input"
dense_76_4831691:  
dense_76_4831693: "
dense_77_4831696:  
dense_77_4831698: 
identity?? dense_76/StatefulPartitionedCall? dense_77/StatefulPartitionedCall?
 dense_76/StatefulPartitionedCallStatefulPartitionedCalldense_76_inputdense_76_4831691dense_76_4831693*
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
E__inference_dense_76_layer_call_and_return_conditional_losses_48315612"
 dense_76/StatefulPartitionedCall?
 dense_77/StatefulPartitionedCallStatefulPartitionedCall)dense_76/StatefulPartitionedCall:output:0dense_77_4831696dense_77_4831698*
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
E__inference_dense_77_layer_call_and_return_conditional_losses_48315972"
 dense_77/StatefulPartitionedCall?
IdentityIdentity)dense_77/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp!^dense_76/StatefulPartitionedCall!^dense_77/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????( : : : : 2D
 dense_76/StatefulPartitionedCall dense_76/StatefulPartitionedCall2D
 dense_77/StatefulPartitionedCall dense_77/StatefulPartitionedCall:[ W
+
_output_shapes
:?????????( 
(
_user_specified_namedense_76_input
?
?
)__inference_model_8_layer_call_fn_4833167
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
D__inference_model_8_layer_call_and_return_conditional_losses_48321602
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
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
)__inference_model_8_layer_call_fn_4832221
input_9
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
StatefulPartitionedCallStatefulPartitionedCallinput_9	aux_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
D__inference_model_8_layer_call_and_return_conditional_losses_48321602
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
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????(
!
_user_specified_name	input_9:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input
??
? 
D__inference_model_8_layer_call_and_return_conditional_losses_4833536
inputs_0
inputs_1V
Dtoken_and_position_embedding_8_embedding_17_embedding_lookup_4833243:( V
Dtoken_and_position_embedding_8_embedding_16_embedding_lookup_4833249: l
Ztransformer_block_8_multi_head_self_attention_8_dense_72_tensordot_readvariableop_resource:  f
Xtransformer_block_8_multi_head_self_attention_8_dense_72_biasadd_readvariableop_resource: l
Ztransformer_block_8_multi_head_self_attention_8_dense_73_tensordot_readvariableop_resource:  f
Xtransformer_block_8_multi_head_self_attention_8_dense_73_biasadd_readvariableop_resource: l
Ztransformer_block_8_multi_head_self_attention_8_dense_74_tensordot_readvariableop_resource:  f
Xtransformer_block_8_multi_head_self_attention_8_dense_74_biasadd_readvariableop_resource: l
Ztransformer_block_8_multi_head_self_attention_8_dense_75_tensordot_readvariableop_resource:  f
Xtransformer_block_8_multi_head_self_attention_8_dense_75_biasadd_readvariableop_resource: ^
Ptransformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource: Z
Ltransformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource: ]
Ktransformer_block_8_sequential_8_dense_76_tensordot_readvariableop_resource:  W
Itransformer_block_8_sequential_8_dense_76_biasadd_readvariableop_resource: ]
Ktransformer_block_8_sequential_8_dense_77_tensordot_readvariableop_resource:  W
Itransformer_block_8_sequential_8_dense_77_biasadd_readvariableop_resource: ^
Ptransformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource: Z
Ltransformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource: ;
)aux_output_matmul_readvariableop_resource: 8
*aux_output_biasadd_readvariableop_resource:9
'dense_78_matmul_readvariableop_resource:@6
(dense_78_biasadd_readvariableop_resource:@9
'dense_79_matmul_readvariableop_resource:@@6
(dense_79_biasadd_readvariableop_resource:@9
'dense_80_matmul_readvariableop_resource:@@6
(dense_80_biasadd_readvariableop_resource:@<
*main_output_matmul_readvariableop_resource:@9
+main_output_biasadd_readvariableop_resource:
identity

identity_1??!aux_output/BiasAdd/ReadVariableOp? aux_output/MatMul/ReadVariableOp?dense_78/BiasAdd/ReadVariableOp?dense_78/MatMul/ReadVariableOp?dense_79/BiasAdd/ReadVariableOp?dense_79/MatMul/ReadVariableOp?dense_80/BiasAdd/ReadVariableOp?dense_80/MatMul/ReadVariableOp?"main_output/BiasAdd/ReadVariableOp?!main_output/MatMul/ReadVariableOp?<token_and_position_embedding_8/embedding_16/embedding_lookup?<token_and_position_embedding_8/embedding_17/embedding_lookup?Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp?Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp?Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp?Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp?Otransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?Otransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?Otransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?Otransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?@transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp?Btransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp?@transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp?Btransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp?
$token_and_position_embedding_8/ShapeShapeinputs_0*
T0*
_output_shapes
:2&
$token_and_position_embedding_8/Shape?
2token_and_position_embedding_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2token_and_position_embedding_8/strided_slice/stack?
4token_and_position_embedding_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_8/strided_slice/stack_1?
4token_and_position_embedding_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_8/strided_slice/stack_2?
,token_and_position_embedding_8/strided_sliceStridedSlice-token_and_position_embedding_8/Shape:output:0;token_and_position_embedding_8/strided_slice/stack:output:0=token_and_position_embedding_8/strided_slice/stack_1:output:0=token_and_position_embedding_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_8/strided_slice?
*token_and_position_embedding_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_8/range/start?
*token_and_position_embedding_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_8/range/delta?
$token_and_position_embedding_8/rangeRange3token_and_position_embedding_8/range/start:output:05token_and_position_embedding_8/strided_slice:output:03token_and_position_embedding_8/range/delta:output:0*#
_output_shapes
:?????????2&
$token_and_position_embedding_8/range?
<token_and_position_embedding_8/embedding_17/embedding_lookupResourceGatherDtoken_and_position_embedding_8_embedding_17_embedding_lookup_4833243-token_and_position_embedding_8/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_17/embedding_lookup/4833243*'
_output_shapes
:????????? *
dtype02>
<token_and_position_embedding_8/embedding_17/embedding_lookup?
Etoken_and_position_embedding_8/embedding_17/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_8/embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_17/embedding_lookup/4833243*'
_output_shapes
:????????? 2G
Etoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity?
Gtoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2I
Gtoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1?
0token_and_position_embedding_8/embedding_16/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????(22
0token_and_position_embedding_8/embedding_16/Cast?
<token_and_position_embedding_8/embedding_16/embedding_lookupResourceGatherDtoken_and_position_embedding_8_embedding_16_embedding_lookup_48332494token_and_position_embedding_8/embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_16/embedding_lookup/4833249*+
_output_shapes
:?????????( *
dtype02>
<token_and_position_embedding_8/embedding_16/embedding_lookup?
Etoken_and_position_embedding_8/embedding_16/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_8/embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@token_and_position_embedding_8/embedding_16/embedding_lookup/4833249*+
_output_shapes
:?????????( 2G
Etoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity?
Gtoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2I
Gtoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1?
"token_and_position_embedding_8/addAddV2Ptoken_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1:output:0Ptoken_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2$
"token_and_position_embedding_8/add?
5transformer_block_8/multi_head_self_attention_8/ShapeShape&token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:27
5transformer_block_8/multi_head_self_attention_8/Shape?
Ctransformer_block_8/multi_head_self_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_8/multi_head_self_attention_8/strided_slice/stack?
Etransformer_block_8/multi_head_self_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Etransformer_block_8/multi_head_self_attention_8/strided_slice/stack_1?
Etransformer_block_8/multi_head_self_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Etransformer_block_8/multi_head_self_attention_8/strided_slice/stack_2?
=transformer_block_8/multi_head_self_attention_8/strided_sliceStridedSlice>transformer_block_8/multi_head_self_attention_8/Shape:output:0Ltransformer_block_8/multi_head_self_attention_8/strided_slice/stack:output:0Ntransformer_block_8/multi_head_self_attention_8/strided_slice/stack_1:output:0Ntransformer_block_8/multi_head_self_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=transformer_block_8/multi_head_self_attention_8/strided_slice?
Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_8_multi_head_self_attention_8_dense_72_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axes?
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/free?
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ShapeShape&token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Shape?
Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/free:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2?
Rtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis?
Mtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axes:output:0[transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1?
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const?
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ProdProdTtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod?
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_1?
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod_1ProdVtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1:output:0Stransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod_1?
Ntransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat/axis?
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concatConcatV2Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/free:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axes:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat?
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/stackPackPtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod:output:0Rtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/stack?
Ltransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/transpose	Transpose&token_and_position_embedding_8/add:z:0Rtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/transpose?
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReshapeReshapePtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/transpose:y:0Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Reshape?
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/MatMulMatMulStransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Reshape:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/MatMul?
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_2?
Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1ConcatV2Ttransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0Stransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_2:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1?
Btransformer_block_8/multi_head_self_attention_8/dense_72/TensordotReshapeStransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/MatMul:product:0Ttransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot?
Otransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_8_multi_head_self_attention_8_dense_72_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?
@transformer_block_8/multi_head_self_attention_8/dense_72/BiasAddBiasAddKtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd?
Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_8_multi_head_self_attention_8_dense_73_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axes?
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/free?
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ShapeShape&token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Shape?
Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/free:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2?
Rtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis?
Mtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axes:output:0[transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1?
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const?
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ProdProdTtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod?
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_1?
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod_1ProdVtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1:output:0Stransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod_1?
Ntransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat/axis?
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concatConcatV2Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/free:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axes:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat?
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/stackPackPtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod:output:0Rtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/stack?
Ltransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/transpose	Transpose&token_and_position_embedding_8/add:z:0Rtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/transpose?
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReshapeReshapePtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/transpose:y:0Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Reshape?
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/MatMulMatMulStransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Reshape:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/MatMul?
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_2?
Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1ConcatV2Ttransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0Stransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_2:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1?
Btransformer_block_8/multi_head_self_attention_8/dense_73/TensordotReshapeStransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/MatMul:product:0Ttransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot?
Otransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_8_multi_head_self_attention_8_dense_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?
@transformer_block_8/multi_head_self_attention_8/dense_73/BiasAddBiasAddKtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd?
Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_8_multi_head_self_attention_8_dense_74_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axes?
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/free?
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ShapeShape&token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Shape?
Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/free:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2?
Rtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis?
Mtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axes:output:0[transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1?
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const?
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ProdProdTtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod?
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_1?
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod_1ProdVtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1:output:0Stransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod_1?
Ntransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat/axis?
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concatConcatV2Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/free:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axes:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat?
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/stackPackPtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod:output:0Rtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/stack?
Ltransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/transpose	Transpose&token_and_position_embedding_8/add:z:0Rtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2N
Ltransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/transpose?
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReshapeReshapePtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/transpose:y:0Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Reshape?
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/MatMulMatMulStransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Reshape:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/MatMul?
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_2?
Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1ConcatV2Ttransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0Stransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_2:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1?
Btransformer_block_8/multi_head_self_attention_8/dense_74/TensordotReshapeStransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/MatMul:product:0Ttransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2D
Btransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot?
Otransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_8_multi_head_self_attention_8_dense_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?
@transformer_block_8/multi_head_self_attention_8/dense_74/BiasAddBiasAddKtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd?
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2A
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/1?
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2A
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/2?
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2A
?transformer_block_8/multi_head_self_attention_8/Reshape/shape/3?
=transformer_block_8/multi_head_self_attention_8/Reshape/shapePackFtransformer_block_8/multi_head_self_attention_8/strided_slice:output:0Htransformer_block_8/multi_head_self_attention_8/Reshape/shape/1:output:0Htransformer_block_8/multi_head_self_attention_8/Reshape/shape/2:output:0Htransformer_block_8/multi_head_self_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_8/multi_head_self_attention_8/Reshape/shape?
7transformer_block_8/multi_head_self_attention_8/ReshapeReshapeItransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd:output:0Ftransformer_block_8/multi_head_self_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????29
7transformer_block_8/multi_head_self_attention_8/Reshape?
>transformer_block_8/multi_head_self_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2@
>transformer_block_8/multi_head_self_attention_8/transpose/perm?
9transformer_block_8/multi_head_self_attention_8/transpose	Transpose@transformer_block_8/multi_head_self_attention_8/Reshape:output:0Gtransformer_block_8/multi_head_self_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_8/multi_head_self_attention_8/transpose?
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/1?
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/2?
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/3?
?transformer_block_8/multi_head_self_attention_8/Reshape_1/shapePackFtransformer_block_8/multi_head_self_attention_8/strided_slice:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/1:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/2:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_8/multi_head_self_attention_8/Reshape_1/shape?
9transformer_block_8/multi_head_self_attention_8/Reshape_1ReshapeItransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd:output:0Htransformer_block_8/multi_head_self_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_8/multi_head_self_attention_8/Reshape_1?
@transformer_block_8/multi_head_self_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_8/multi_head_self_attention_8/transpose_1/perm?
;transformer_block_8/multi_head_self_attention_8/transpose_1	TransposeBtransformer_block_8/multi_head_self_attention_8/Reshape_1:output:0Itransformer_block_8/multi_head_self_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_8/multi_head_self_attention_8/transpose_1?
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/1?
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/2?
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/3?
?transformer_block_8/multi_head_self_attention_8/Reshape_2/shapePackFtransformer_block_8/multi_head_self_attention_8/strided_slice:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/1:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/2:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_8/multi_head_self_attention_8/Reshape_2/shape?
9transformer_block_8/multi_head_self_attention_8/Reshape_2ReshapeItransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd:output:0Htransformer_block_8/multi_head_self_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2;
9transformer_block_8/multi_head_self_attention_8/Reshape_2?
@transformer_block_8/multi_head_self_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_8/multi_head_self_attention_8/transpose_2/perm?
;transformer_block_8/multi_head_self_attention_8/transpose_2	TransposeBtransformer_block_8/multi_head_self_attention_8/Reshape_2:output:0Itransformer_block_8/multi_head_self_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_8/multi_head_self_attention_8/transpose_2?
6transformer_block_8/multi_head_self_attention_8/MatMulBatchMatMulV2=transformer_block_8/multi_head_self_attention_8/transpose:y:0?transformer_block_8/multi_head_self_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(28
6transformer_block_8/multi_head_self_attention_8/MatMul?
7transformer_block_8/multi_head_self_attention_8/Shape_1Shape?transformer_block_8/multi_head_self_attention_8/transpose_1:y:0*
T0*
_output_shapes
:29
7transformer_block_8/multi_head_self_attention_8/Shape_1?
Etransformer_block_8/multi_head_self_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2G
Etransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack?
Gtransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2I
Gtransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_1?
Gtransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_2?
?transformer_block_8/multi_head_self_attention_8/strided_slice_1StridedSlice@transformer_block_8/multi_head_self_attention_8/Shape_1:output:0Ntransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack:output:0Ptransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_1:output:0Ptransformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?transformer_block_8/multi_head_self_attention_8/strided_slice_1?
4transformer_block_8/multi_head_self_attention_8/CastCastHtransformer_block_8/multi_head_self_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 26
4transformer_block_8/multi_head_self_attention_8/Cast?
4transformer_block_8/multi_head_self_attention_8/SqrtSqrt8transformer_block_8/multi_head_self_attention_8/Cast:y:0*
T0*
_output_shapes
: 26
4transformer_block_8/multi_head_self_attention_8/Sqrt?
7transformer_block_8/multi_head_self_attention_8/truedivRealDiv?transformer_block_8/multi_head_self_attention_8/MatMul:output:08transformer_block_8/multi_head_self_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????29
7transformer_block_8/multi_head_self_attention_8/truediv?
7transformer_block_8/multi_head_self_attention_8/SoftmaxSoftmax;transformer_block_8/multi_head_self_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????29
7transformer_block_8/multi_head_self_attention_8/Softmax?
8transformer_block_8/multi_head_self_attention_8/MatMul_1BatchMatMulV2Atransformer_block_8/multi_head_self_attention_8/Softmax:softmax:0?transformer_block_8/multi_head_self_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2:
8transformer_block_8/multi_head_self_attention_8/MatMul_1?
@transformer_block_8/multi_head_self_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2B
@transformer_block_8/multi_head_self_attention_8/transpose_3/perm?
;transformer_block_8/multi_head_self_attention_8/transpose_3	TransposeAtransformer_block_8/multi_head_self_attention_8/MatMul_1:output:0Itransformer_block_8/multi_head_self_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2=
;transformer_block_8/multi_head_self_attention_8/transpose_3?
Atransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/1?
Atransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/2?
?transformer_block_8/multi_head_self_attention_8/Reshape_3/shapePackFtransformer_block_8/multi_head_self_attention_8/strided_slice:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/1:output:0Jtransformer_block_8/multi_head_self_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2A
?transformer_block_8/multi_head_self_attention_8/Reshape_3/shape?
9transformer_block_8/multi_head_self_attention_8/Reshape_3Reshape?transformer_block_8/multi_head_self_attention_8/transpose_3:y:0Htransformer_block_8/multi_head_self_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2;
9transformer_block_8/multi_head_self_attention_8/Reshape_3?
Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpReadVariableOpZtransformer_block_8_multi_head_self_attention_8_dense_75_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02S
Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2I
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axes?
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2I
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/free?
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ShapeShapeBtransformer_block_8/multi_head_self_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Shape?
Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/free:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2?
Rtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis?
Mtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1GatherV2Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axes:output:0[transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2O
Mtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1?
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2J
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const?
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ProdProdTtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const:output:0*
T0*
_output_shapes
: 2I
Gtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod?
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_1?
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod_1ProdVtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1:output:0Stransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2K
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod_1?
Ntransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2P
Ntransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat/axis?
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concatConcatV2Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/free:output:0Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axes:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2K
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat?
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/stackPackPtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod:output:0Rtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2J
Htransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/stack?
Ltransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/transpose	TransposeBtransformer_block_8/multi_head_self_attention_8/Reshape_3:output:0Rtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2N
Ltransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/transpose?
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReshapeReshapePtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/transpose:y:0Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2L
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Reshape?
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/MatMulMatMulStransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Reshape:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2K
Itransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/MatMul?
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_2?
Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2R
Ptransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1/axis?
Ktransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1ConcatV2Ttransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0Stransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_2:output:0Ytransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2M
Ktransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1?
Btransformer_block_8/multi_head_self_attention_8/dense_75/TensordotReshapeStransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/MatMul:product:0Ttransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 2D
Btransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot?
Otransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpReadVariableOpXtransformer_block_8_multi_head_self_attention_8_dense_75_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Q
Otransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?
@transformer_block_8/multi_head_self_attention_8/dense_75/BiasAddBiasAddKtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot:output:0Wtransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2B
@transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd?
'transformer_block_8/dropout_16/IdentityIdentityItransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 2)
'transformer_block_8/dropout_16/Identity?
transformer_block_8/addAddV2&token_and_position_embedding_8/add:z:00transformer_block_8/dropout_16/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_8/add?
Itransformer_block_8/layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_8/layer_normalization_16/moments/mean/reduction_indices?
7transformer_block_8/layer_normalization_16/moments/meanMeantransformer_block_8/add:z:0Rtransformer_block_8/layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(29
7transformer_block_8/layer_normalization_16/moments/mean?
?transformer_block_8/layer_normalization_16/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2A
?transformer_block_8/layer_normalization_16/moments/StopGradient?
Dtransformer_block_8/layer_normalization_16/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add:z:0Htransformer_block_8/layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2F
Dtransformer_block_8/layer_normalization_16/moments/SquaredDifference?
Mtransformer_block_8/layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_8/layer_normalization_16/moments/variance/reduction_indices?
;transformer_block_8/layer_normalization_16/moments/varianceMeanHtransformer_block_8/layer_normalization_16/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2=
;transformer_block_8/layer_normalization_16/moments/variance?
:transformer_block_8/layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:transformer_block_8/layer_normalization_16/batchnorm/add/y?
8transformer_block_8/layer_normalization_16/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_16/moments/variance:output:0Ctransformer_block_8/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2:
8transformer_block_8/layer_normalization_16/batchnorm/add?
:transformer_block_8/layer_normalization_16/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2<
:transformer_block_8/layer_normalization_16/batchnorm/Rsqrt?
Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp?
8transformer_block_8/layer_normalization_16/batchnorm/mulMul>transformer_block_8/layer_normalization_16/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2:
8transformer_block_8/layer_normalization_16/batchnorm/mul?
:transformer_block_8/layer_normalization_16/batchnorm/mul_1Multransformer_block_8/add:z:0<transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_16/batchnorm/mul_1?
:transformer_block_8/layer_normalization_16/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_16/moments/mean:output:0<transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_16/batchnorm/mul_2?
Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp?
8transformer_block_8/layer_normalization_16/batchnorm/subSubKtransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2:
8transformer_block_8/layer_normalization_16/batchnorm/sub?
:transformer_block_8/layer_normalization_16/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_16/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_16/batchnorm/add_1?
Btransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_8_sequential_8_dense_76_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02D
Btransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp?
8transformer_block_8/sequential_8/dense_76/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_8/sequential_8/dense_76/Tensordot/axes?
8transformer_block_8/sequential_8/dense_76/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_8/sequential_8/dense_76/Tensordot/free?
9transformer_block_8/sequential_8/dense_76/Tensordot/ShapeShape>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:2;
9transformer_block_8/sequential_8/dense_76/Tensordot/Shape?
Atransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2/axis?
<transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2GatherV2Btransformer_block_8/sequential_8/dense_76/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_76/Tensordot/free:output:0Jtransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2?
Ctransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1/axis?
>transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1GatherV2Btransformer_block_8/sequential_8/dense_76/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_76/Tensordot/axes:output:0Ltransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1?
9transformer_block_8/sequential_8/dense_76/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_8/sequential_8/dense_76/Tensordot/Const?
8transformer_block_8/sequential_8/dense_76/Tensordot/ProdProdEtransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2:output:0Btransformer_block_8/sequential_8/dense_76/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_8/sequential_8/dense_76/Tensordot/Prod?
;transformer_block_8/sequential_8/dense_76/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_8/sequential_8/dense_76/Tensordot/Const_1?
:transformer_block_8/sequential_8/dense_76/Tensordot/Prod_1ProdGtransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1:output:0Dtransformer_block_8/sequential_8/dense_76/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_8/sequential_8/dense_76/Tensordot/Prod_1?
?transformer_block_8/sequential_8/dense_76/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_8/sequential_8/dense_76/Tensordot/concat/axis?
:transformer_block_8/sequential_8/dense_76/Tensordot/concatConcatV2Atransformer_block_8/sequential_8/dense_76/Tensordot/free:output:0Atransformer_block_8/sequential_8/dense_76/Tensordot/axes:output:0Htransformer_block_8/sequential_8/dense_76/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_76/Tensordot/concat?
9transformer_block_8/sequential_8/dense_76/Tensordot/stackPackAtransformer_block_8/sequential_8/dense_76/Tensordot/Prod:output:0Ctransformer_block_8/sequential_8/dense_76/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_8/sequential_8/dense_76/Tensordot/stack?
=transformer_block_8/sequential_8/dense_76/Tensordot/transpose	Transpose>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0Ctransformer_block_8/sequential_8/dense_76/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2?
=transformer_block_8/sequential_8/dense_76/Tensordot/transpose?
;transformer_block_8/sequential_8/dense_76/Tensordot/ReshapeReshapeAtransformer_block_8/sequential_8/dense_76/Tensordot/transpose:y:0Btransformer_block_8/sequential_8/dense_76/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_8/sequential_8/dense_76/Tensordot/Reshape?
:transformer_block_8/sequential_8/dense_76/Tensordot/MatMulMatMulDtransformer_block_8/sequential_8/dense_76/Tensordot/Reshape:output:0Jtransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_8/sequential_8/dense_76/Tensordot/MatMul?
;transformer_block_8/sequential_8/dense_76/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_8/sequential_8/dense_76/Tensordot/Const_2?
Atransformer_block_8/sequential_8/dense_76/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_8/sequential_8/dense_76/Tensordot/concat_1/axis?
<transformer_block_8/sequential_8/dense_76/Tensordot/concat_1ConcatV2Etransformer_block_8/sequential_8/dense_76/Tensordot/GatherV2:output:0Dtransformer_block_8/sequential_8/dense_76/Tensordot/Const_2:output:0Jtransformer_block_8/sequential_8/dense_76/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_8/sequential_8/dense_76/Tensordot/concat_1?
3transformer_block_8/sequential_8/dense_76/TensordotReshapeDtransformer_block_8/sequential_8/dense_76/Tensordot/MatMul:product:0Etransformer_block_8/sequential_8/dense_76/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 25
3transformer_block_8/sequential_8/dense_76/Tensordot?
@transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_8_sequential_8_dense_76_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp?
1transformer_block_8/sequential_8/dense_76/BiasAddBiasAdd<transformer_block_8/sequential_8/dense_76/Tensordot:output:0Htransformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 23
1transformer_block_8/sequential_8/dense_76/BiasAdd?
.transformer_block_8/sequential_8/dense_76/ReluRelu:transformer_block_8/sequential_8/dense_76/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 20
.transformer_block_8/sequential_8/dense_76/Relu?
Btransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOpReadVariableOpKtransformer_block_8_sequential_8_dense_77_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02D
Btransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp?
8transformer_block_8/sequential_8/dense_77/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8transformer_block_8/sequential_8/dense_77/Tensordot/axes?
8transformer_block_8/sequential_8/dense_77/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8transformer_block_8/sequential_8/dense_77/Tensordot/free?
9transformer_block_8/sequential_8/dense_77/Tensordot/ShapeShape<transformer_block_8/sequential_8/dense_76/Relu:activations:0*
T0*
_output_shapes
:2;
9transformer_block_8/sequential_8/dense_77/Tensordot/Shape?
Atransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2/axis?
<transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2GatherV2Btransformer_block_8/sequential_8/dense_77/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_77/Tensordot/free:output:0Jtransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2?
Ctransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Ctransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1/axis?
>transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1GatherV2Btransformer_block_8/sequential_8/dense_77/Tensordot/Shape:output:0Atransformer_block_8/sequential_8/dense_77/Tensordot/axes:output:0Ltransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1?
9transformer_block_8/sequential_8/dense_77/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block_8/sequential_8/dense_77/Tensordot/Const?
8transformer_block_8/sequential_8/dense_77/Tensordot/ProdProdEtransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2:output:0Btransformer_block_8/sequential_8/dense_77/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8transformer_block_8/sequential_8/dense_77/Tensordot/Prod?
;transformer_block_8/sequential_8/dense_77/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_8/sequential_8/dense_77/Tensordot/Const_1?
:transformer_block_8/sequential_8/dense_77/Tensordot/Prod_1ProdGtransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1:output:0Dtransformer_block_8/sequential_8/dense_77/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:transformer_block_8/sequential_8/dense_77/Tensordot/Prod_1?
?transformer_block_8/sequential_8/dense_77/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?transformer_block_8/sequential_8/dense_77/Tensordot/concat/axis?
:transformer_block_8/sequential_8/dense_77/Tensordot/concatConcatV2Atransformer_block_8/sequential_8/dense_77/Tensordot/free:output:0Atransformer_block_8/sequential_8/dense_77/Tensordot/axes:output:0Htransformer_block_8/sequential_8/dense_77/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_77/Tensordot/concat?
9transformer_block_8/sequential_8/dense_77/Tensordot/stackPackAtransformer_block_8/sequential_8/dense_77/Tensordot/Prod:output:0Ctransformer_block_8/sequential_8/dense_77/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block_8/sequential_8/dense_77/Tensordot/stack?
=transformer_block_8/sequential_8/dense_77/Tensordot/transpose	Transpose<transformer_block_8/sequential_8/dense_76/Relu:activations:0Ctransformer_block_8/sequential_8/dense_77/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2?
=transformer_block_8/sequential_8/dense_77/Tensordot/transpose?
;transformer_block_8/sequential_8/dense_77/Tensordot/ReshapeReshapeAtransformer_block_8/sequential_8/dense_77/Tensordot/transpose:y:0Btransformer_block_8/sequential_8/dense_77/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2=
;transformer_block_8/sequential_8/dense_77/Tensordot/Reshape?
:transformer_block_8/sequential_8/dense_77/Tensordot/MatMulMatMulDtransformer_block_8/sequential_8/dense_77/Tensordot/Reshape:output:0Jtransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2<
:transformer_block_8/sequential_8/dense_77/Tensordot/MatMul?
;transformer_block_8/sequential_8/dense_77/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block_8/sequential_8/dense_77/Tensordot/Const_2?
Atransformer_block_8/sequential_8/dense_77/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Atransformer_block_8/sequential_8/dense_77/Tensordot/concat_1/axis?
<transformer_block_8/sequential_8/dense_77/Tensordot/concat_1ConcatV2Etransformer_block_8/sequential_8/dense_77/Tensordot/GatherV2:output:0Dtransformer_block_8/sequential_8/dense_77/Tensordot/Const_2:output:0Jtransformer_block_8/sequential_8/dense_77/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<transformer_block_8/sequential_8/dense_77/Tensordot/concat_1?
3transformer_block_8/sequential_8/dense_77/TensordotReshapeDtransformer_block_8/sequential_8/dense_77/Tensordot/MatMul:product:0Etransformer_block_8/sequential_8/dense_77/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 25
3transformer_block_8/sequential_8/dense_77/Tensordot?
@transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOpReadVariableOpItransformer_block_8_sequential_8_dense_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02B
@transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp?
1transformer_block_8/sequential_8/dense_77/BiasAddBiasAdd<transformer_block_8/sequential_8/dense_77/Tensordot:output:0Htransformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 23
1transformer_block_8/sequential_8/dense_77/BiasAdd?
'transformer_block_8/dropout_17/IdentityIdentity:transformer_block_8/sequential_8/dense_77/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2)
'transformer_block_8/dropout_17/Identity?
transformer_block_8/add_1AddV2>transformer_block_8/layer_normalization_16/batchnorm/add_1:z:00transformer_block_8/dropout_17/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
transformer_block_8/add_1?
Itransformer_block_8/layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_8/layer_normalization_17/moments/mean/reduction_indices?
7transformer_block_8/layer_normalization_17/moments/meanMeantransformer_block_8/add_1:z:0Rtransformer_block_8/layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(29
7transformer_block_8/layer_normalization_17/moments/mean?
?transformer_block_8/layer_normalization_17/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2A
?transformer_block_8/layer_normalization_17/moments/StopGradient?
Dtransformer_block_8/layer_normalization_17/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add_1:z:0Htransformer_block_8/layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2F
Dtransformer_block_8/layer_normalization_17/moments/SquaredDifference?
Mtransformer_block_8/layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_8/layer_normalization_17/moments/variance/reduction_indices?
;transformer_block_8/layer_normalization_17/moments/varianceMeanHtransformer_block_8/layer_normalization_17/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2=
;transformer_block_8/layer_normalization_17/moments/variance?
:transformer_block_8/layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:transformer_block_8/layer_normalization_17/batchnorm/add/y?
8transformer_block_8/layer_normalization_17/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_17/moments/variance:output:0Ctransformer_block_8/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2:
8transformer_block_8/layer_normalization_17/batchnorm/add?
:transformer_block_8/layer_normalization_17/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2<
:transformer_block_8/layer_normalization_17/batchnorm/Rsqrt?
Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02I
Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp?
8transformer_block_8/layer_normalization_17/batchnorm/mulMul>transformer_block_8/layer_normalization_17/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2:
8transformer_block_8/layer_normalization_17/batchnorm/mul?
:transformer_block_8/layer_normalization_17/batchnorm/mul_1Multransformer_block_8/add_1:z:0<transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_17/batchnorm/mul_1?
:transformer_block_8/layer_normalization_17/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_17/moments/mean:output:0<transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_17/batchnorm/mul_2?
Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02E
Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp?
8transformer_block_8/layer_normalization_17/batchnorm/subSubKtransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2:
8transformer_block_8/layer_normalization_17/batchnorm/sub?
:transformer_block_8/layer_normalization_17/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_17/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2<
:transformer_block_8/layer_normalization_17/batchnorm/add_1?
1global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_8/Mean/reduction_indices?
global_average_pooling1d_8/MeanMean>transformer_block_8/layer_normalization_17/batchnorm/add_1:z:0:global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2!
global_average_pooling1d_8/Mean?
 aux_output/MatMul/ReadVariableOpReadVariableOp)aux_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02"
 aux_output/MatMul/ReadVariableOp?
aux_output/MatMulMatMul(global_average_pooling1d_8/Mean:output:0(aux_output/MatMul/ReadVariableOp:value:0*
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
concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_8/concat/axis?
concatenate_8/concatConcatV2aux_output/Sigmoid:y:0inputs_1"concatenate_8/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate_8/concat?
dense_78/MatMul/ReadVariableOpReadVariableOp'dense_78_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_78/MatMul/ReadVariableOp?
dense_78/MatMulMatMulconcatenate_8/concat:output:0&dense_78/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_78/MatMul?
dense_78/BiasAdd/ReadVariableOpReadVariableOp(dense_78_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_78/BiasAdd/ReadVariableOp?
dense_78/BiasAddBiasAdddense_78/MatMul:product:0'dense_78/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_78/BiasAdds
dense_78/ReluReludense_78/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_78/Relu?
dense_79/MatMul/ReadVariableOpReadVariableOp'dense_79_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_79/MatMul/ReadVariableOp?
dense_79/MatMulMatMuldense_78/Relu:activations:0&dense_79/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_79/MatMul?
dense_79/BiasAdd/ReadVariableOpReadVariableOp(dense_79_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_79/BiasAdd/ReadVariableOp?
dense_79/BiasAddBiasAdddense_79/MatMul:product:0'dense_79/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_79/BiasAdds
dense_79/ReluReludense_79/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_79/Relu?
dense_80/MatMul/ReadVariableOpReadVariableOp'dense_80_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_80/MatMul/ReadVariableOp?
dense_80/MatMulMatMuldense_79/Relu:activations:0&dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_80/MatMul?
dense_80/BiasAdd/ReadVariableOpReadVariableOp(dense_80_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_80/BiasAdd/ReadVariableOp?
dense_80/BiasAddBiasAdddense_80/MatMul:product:0'dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_80/BiasAdds
dense_80/ReluReludense_80/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_80/Relu?
!main_output/MatMul/ReadVariableOpReadVariableOp*main_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!main_output/MatMul/ReadVariableOp?
main_output/MatMulMatMuldense_80/Relu:activations:0)main_output/MatMul/ReadVariableOp:value:0*
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
NoOpNoOp"^aux_output/BiasAdd/ReadVariableOp!^aux_output/MatMul/ReadVariableOp ^dense_78/BiasAdd/ReadVariableOp^dense_78/MatMul/ReadVariableOp ^dense_79/BiasAdd/ReadVariableOp^dense_79/MatMul/ReadVariableOp ^dense_80/BiasAdd/ReadVariableOp^dense_80/MatMul/ReadVariableOp#^main_output/BiasAdd/ReadVariableOp"^main_output/MatMul/ReadVariableOp=^token_and_position_embedding_8/embedding_16/embedding_lookup=^token_and_position_embedding_8/embedding_17/embedding_lookupD^transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpD^transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpP^transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpR^transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpP^transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpR^transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpP^transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpR^transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpP^transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpR^transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpA^transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOpC^transformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOpA^transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOpC^transformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!aux_output/BiasAdd/ReadVariableOp!aux_output/BiasAdd/ReadVariableOp2D
 aux_output/MatMul/ReadVariableOp aux_output/MatMul/ReadVariableOp2B
dense_78/BiasAdd/ReadVariableOpdense_78/BiasAdd/ReadVariableOp2@
dense_78/MatMul/ReadVariableOpdense_78/MatMul/ReadVariableOp2B
dense_79/BiasAdd/ReadVariableOpdense_79/BiasAdd/ReadVariableOp2@
dense_79/MatMul/ReadVariableOpdense_79/MatMul/ReadVariableOp2B
dense_80/BiasAdd/ReadVariableOpdense_80/BiasAdd/ReadVariableOp2@
dense_80/MatMul/ReadVariableOpdense_80/MatMul/ReadVariableOp2H
"main_output/BiasAdd/ReadVariableOp"main_output/BiasAdd/ReadVariableOp2F
!main_output/MatMul/ReadVariableOp!main_output/MatMul/ReadVariableOp2|
<token_and_position_embedding_8/embedding_16/embedding_lookup<token_and_position_embedding_8/embedding_16/embedding_lookup2|
<token_and_position_embedding_8/embedding_17/embedding_lookup<token_and_position_embedding_8/embedding_17/embedding_lookup2?
Ctransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp2?
Gtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp2?
Ctransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp2?
Gtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp2?
Otransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpOtransformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp2?
Qtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp2?
Otransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpOtransformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp2?
Qtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp2?
Otransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpOtransformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp2?
Qtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp2?
Otransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpOtransformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp2?
Qtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpQtransformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp2?
@transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp@transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp2?
Btransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOpBtransformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp2?
@transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp@transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp2?
Btransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOpBtransformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
??
?5
 __inference__traced_save_4835140
file_prefix0
,savev2_aux_output_kernel_read_readvariableop.
*savev2_aux_output_bias_read_readvariableop.
*savev2_dense_78_kernel_read_readvariableop,
(savev2_dense_78_bias_read_readvariableop.
*savev2_dense_79_kernel_read_readvariableop,
(savev2_dense_79_bias_read_readvariableop.
*savev2_dense_80_kernel_read_readvariableop,
(savev2_dense_80_bias_read_readvariableop1
-savev2_main_output_kernel_read_readvariableop/
+savev2_main_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopU
Qsavev2_token_and_position_embedding_8_embedding_16_embeddings_read_readvariableopU
Qsavev2_token_and_position_embedding_8_embedding_17_embeddings_read_readvariableop^
Zsavev2_transformer_block_8_multi_head_self_attention_8_dense_72_kernel_read_readvariableop\
Xsavev2_transformer_block_8_multi_head_self_attention_8_dense_72_bias_read_readvariableop^
Zsavev2_transformer_block_8_multi_head_self_attention_8_dense_73_kernel_read_readvariableop\
Xsavev2_transformer_block_8_multi_head_self_attention_8_dense_73_bias_read_readvariableop^
Zsavev2_transformer_block_8_multi_head_self_attention_8_dense_74_kernel_read_readvariableop\
Xsavev2_transformer_block_8_multi_head_self_attention_8_dense_74_bias_read_readvariableop^
Zsavev2_transformer_block_8_multi_head_self_attention_8_dense_75_kernel_read_readvariableop\
Xsavev2_transformer_block_8_multi_head_self_attention_8_dense_75_bias_read_readvariableop.
*savev2_dense_76_kernel_read_readvariableop,
(savev2_dense_76_bias_read_readvariableop.
*savev2_dense_77_kernel_read_readvariableop,
(savev2_dense_77_bias_read_readvariableopO
Ksavev2_transformer_block_8_layer_normalization_16_gamma_read_readvariableopN
Jsavev2_transformer_block_8_layer_normalization_16_beta_read_readvariableopO
Ksavev2_transformer_block_8_layer_normalization_17_gamma_read_readvariableopN
Jsavev2_transformer_block_8_layer_normalization_17_beta_read_readvariableop$
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
1savev2_adam_dense_78_kernel_m_read_readvariableop3
/savev2_adam_dense_78_bias_m_read_readvariableop5
1savev2_adam_dense_79_kernel_m_read_readvariableop3
/savev2_adam_dense_79_bias_m_read_readvariableop5
1savev2_adam_dense_80_kernel_m_read_readvariableop3
/savev2_adam_dense_80_bias_m_read_readvariableop8
4savev2_adam_main_output_kernel_m_read_readvariableop6
2savev2_adam_main_output_bias_m_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_8_embedding_16_embeddings_m_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_8_embedding_17_embeddings_m_read_readvariableope
asavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_72_kernel_m_read_readvariableopc
_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_72_bias_m_read_readvariableope
asavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_73_kernel_m_read_readvariableopc
_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_73_bias_m_read_readvariableope
asavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_74_kernel_m_read_readvariableopc
_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_74_bias_m_read_readvariableope
asavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_75_kernel_m_read_readvariableopc
_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_75_bias_m_read_readvariableop5
1savev2_adam_dense_76_kernel_m_read_readvariableop3
/savev2_adam_dense_76_bias_m_read_readvariableop5
1savev2_adam_dense_77_kernel_m_read_readvariableop3
/savev2_adam_dense_77_bias_m_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_16_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_16_beta_m_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_17_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_17_beta_m_read_readvariableop7
3savev2_adam_aux_output_kernel_v_read_readvariableop5
1savev2_adam_aux_output_bias_v_read_readvariableop5
1savev2_adam_dense_78_kernel_v_read_readvariableop3
/savev2_adam_dense_78_bias_v_read_readvariableop5
1savev2_adam_dense_79_kernel_v_read_readvariableop3
/savev2_adam_dense_79_bias_v_read_readvariableop5
1savev2_adam_dense_80_kernel_v_read_readvariableop3
/savev2_adam_dense_80_bias_v_read_readvariableop8
4savev2_adam_main_output_kernel_v_read_readvariableop6
2savev2_adam_main_output_bias_v_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_8_embedding_16_embeddings_v_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_8_embedding_17_embeddings_v_read_readvariableope
asavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_72_kernel_v_read_readvariableopc
_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_72_bias_v_read_readvariableope
asavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_73_kernel_v_read_readvariableopc
_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_73_bias_v_read_readvariableope
asavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_74_kernel_v_read_readvariableopc
_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_74_bias_v_read_readvariableope
asavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_75_kernel_v_read_readvariableopc
_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_75_bias_v_read_readvariableop5
1savev2_adam_dense_76_kernel_v_read_readvariableop3
/savev2_adam_dense_76_bias_v_read_readvariableop5
1savev2_adam_dense_77_kernel_v_read_readvariableop3
/savev2_adam_dense_77_bias_v_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_16_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_16_beta_v_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_17_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_17_beta_v_read_readvariableop
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
SaveV2/shape_and_slices?4
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_aux_output_kernel_read_readvariableop*savev2_aux_output_bias_read_readvariableop*savev2_dense_78_kernel_read_readvariableop(savev2_dense_78_bias_read_readvariableop*savev2_dense_79_kernel_read_readvariableop(savev2_dense_79_bias_read_readvariableop*savev2_dense_80_kernel_read_readvariableop(savev2_dense_80_bias_read_readvariableop-savev2_main_output_kernel_read_readvariableop+savev2_main_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopQsavev2_token_and_position_embedding_8_embedding_16_embeddings_read_readvariableopQsavev2_token_and_position_embedding_8_embedding_17_embeddings_read_readvariableopZsavev2_transformer_block_8_multi_head_self_attention_8_dense_72_kernel_read_readvariableopXsavev2_transformer_block_8_multi_head_self_attention_8_dense_72_bias_read_readvariableopZsavev2_transformer_block_8_multi_head_self_attention_8_dense_73_kernel_read_readvariableopXsavev2_transformer_block_8_multi_head_self_attention_8_dense_73_bias_read_readvariableopZsavev2_transformer_block_8_multi_head_self_attention_8_dense_74_kernel_read_readvariableopXsavev2_transformer_block_8_multi_head_self_attention_8_dense_74_bias_read_readvariableopZsavev2_transformer_block_8_multi_head_self_attention_8_dense_75_kernel_read_readvariableopXsavev2_transformer_block_8_multi_head_self_attention_8_dense_75_bias_read_readvariableop*savev2_dense_76_kernel_read_readvariableop(savev2_dense_76_bias_read_readvariableop*savev2_dense_77_kernel_read_readvariableop(savev2_dense_77_bias_read_readvariableopKsavev2_transformer_block_8_layer_normalization_16_gamma_read_readvariableopJsavev2_transformer_block_8_layer_normalization_16_beta_read_readvariableopKsavev2_transformer_block_8_layer_normalization_17_gamma_read_readvariableopJsavev2_transformer_block_8_layer_normalization_17_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop3savev2_adam_aux_output_kernel_m_read_readvariableop1savev2_adam_aux_output_bias_m_read_readvariableop1savev2_adam_dense_78_kernel_m_read_readvariableop/savev2_adam_dense_78_bias_m_read_readvariableop1savev2_adam_dense_79_kernel_m_read_readvariableop/savev2_adam_dense_79_bias_m_read_readvariableop1savev2_adam_dense_80_kernel_m_read_readvariableop/savev2_adam_dense_80_bias_m_read_readvariableop4savev2_adam_main_output_kernel_m_read_readvariableop2savev2_adam_main_output_bias_m_read_readvariableopXsavev2_adam_token_and_position_embedding_8_embedding_16_embeddings_m_read_readvariableopXsavev2_adam_token_and_position_embedding_8_embedding_17_embeddings_m_read_readvariableopasavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_72_kernel_m_read_readvariableop_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_72_bias_m_read_readvariableopasavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_73_kernel_m_read_readvariableop_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_73_bias_m_read_readvariableopasavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_74_kernel_m_read_readvariableop_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_74_bias_m_read_readvariableopasavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_75_kernel_m_read_readvariableop_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_75_bias_m_read_readvariableop1savev2_adam_dense_76_kernel_m_read_readvariableop/savev2_adam_dense_76_bias_m_read_readvariableop1savev2_adam_dense_77_kernel_m_read_readvariableop/savev2_adam_dense_77_bias_m_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_16_gamma_m_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_16_beta_m_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_17_gamma_m_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_17_beta_m_read_readvariableop3savev2_adam_aux_output_kernel_v_read_readvariableop1savev2_adam_aux_output_bias_v_read_readvariableop1savev2_adam_dense_78_kernel_v_read_readvariableop/savev2_adam_dense_78_bias_v_read_readvariableop1savev2_adam_dense_79_kernel_v_read_readvariableop/savev2_adam_dense_79_bias_v_read_readvariableop1savev2_adam_dense_80_kernel_v_read_readvariableop/savev2_adam_dense_80_bias_v_read_readvariableop4savev2_adam_main_output_kernel_v_read_readvariableop2savev2_adam_main_output_bias_v_read_readvariableopXsavev2_adam_token_and_position_embedding_8_embedding_16_embeddings_v_read_readvariableopXsavev2_adam_token_and_position_embedding_8_embedding_17_embeddings_v_read_readvariableopasavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_72_kernel_v_read_readvariableop_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_72_bias_v_read_readvariableopasavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_73_kernel_v_read_readvariableop_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_73_bias_v_read_readvariableopasavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_74_kernel_v_read_readvariableop_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_74_bias_v_read_readvariableopasavev2_adam_transformer_block_8_multi_head_self_attention_8_dense_75_kernel_v_read_readvariableop_savev2_adam_transformer_block_8_multi_head_self_attention_8_dense_75_bias_v_read_readvariableop1savev2_adam_dense_76_kernel_v_read_readvariableop/savev2_adam_dense_76_bias_v_read_readvariableop1savev2_adam_dense_77_kernel_v_read_readvariableop/savev2_adam_dense_77_bias_v_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_16_gamma_v_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_16_beta_v_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_17_gamma_v_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_17_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
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
??
?#
"__inference__wrapped_model_4831523
input_9
	aux_input^
Lmodel_8_token_and_position_embedding_8_embedding_17_embedding_lookup_4831230:( ^
Lmodel_8_token_and_position_embedding_8_embedding_16_embedding_lookup_4831236: t
bmodel_8_transformer_block_8_multi_head_self_attention_8_dense_72_tensordot_readvariableop_resource:  n
`model_8_transformer_block_8_multi_head_self_attention_8_dense_72_biasadd_readvariableop_resource: t
bmodel_8_transformer_block_8_multi_head_self_attention_8_dense_73_tensordot_readvariableop_resource:  n
`model_8_transformer_block_8_multi_head_self_attention_8_dense_73_biasadd_readvariableop_resource: t
bmodel_8_transformer_block_8_multi_head_self_attention_8_dense_74_tensordot_readvariableop_resource:  n
`model_8_transformer_block_8_multi_head_self_attention_8_dense_74_biasadd_readvariableop_resource: t
bmodel_8_transformer_block_8_multi_head_self_attention_8_dense_75_tensordot_readvariableop_resource:  n
`model_8_transformer_block_8_multi_head_self_attention_8_dense_75_biasadd_readvariableop_resource: f
Xmodel_8_transformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource: b
Tmodel_8_transformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource: e
Smodel_8_transformer_block_8_sequential_8_dense_76_tensordot_readvariableop_resource:  _
Qmodel_8_transformer_block_8_sequential_8_dense_76_biasadd_readvariableop_resource: e
Smodel_8_transformer_block_8_sequential_8_dense_77_tensordot_readvariableop_resource:  _
Qmodel_8_transformer_block_8_sequential_8_dense_77_biasadd_readvariableop_resource: f
Xmodel_8_transformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource: b
Tmodel_8_transformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource: C
1model_8_aux_output_matmul_readvariableop_resource: @
2model_8_aux_output_biasadd_readvariableop_resource:A
/model_8_dense_78_matmul_readvariableop_resource:@>
0model_8_dense_78_biasadd_readvariableop_resource:@A
/model_8_dense_79_matmul_readvariableop_resource:@@>
0model_8_dense_79_biasadd_readvariableop_resource:@A
/model_8_dense_80_matmul_readvariableop_resource:@@>
0model_8_dense_80_biasadd_readvariableop_resource:@D
2model_8_main_output_matmul_readvariableop_resource:@A
3model_8_main_output_biasadd_readvariableop_resource:
identity

identity_1??)model_8/aux_output/BiasAdd/ReadVariableOp?(model_8/aux_output/MatMul/ReadVariableOp?'model_8/dense_78/BiasAdd/ReadVariableOp?&model_8/dense_78/MatMul/ReadVariableOp?'model_8/dense_79/BiasAdd/ReadVariableOp?&model_8/dense_79/MatMul/ReadVariableOp?'model_8/dense_80/BiasAdd/ReadVariableOp?&model_8/dense_80/MatMul/ReadVariableOp?*model_8/main_output/BiasAdd/ReadVariableOp?)model_8/main_output/MatMul/ReadVariableOp?Dmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup?Dmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup?Kmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp?Omodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp?Kmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp?Omodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp?Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?Hmodel_8/transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp?Jmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp?Hmodel_8/transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp?Jmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp?
,model_8/token_and_position_embedding_8/ShapeShapeinput_9*
T0*
_output_shapes
:2.
,model_8/token_and_position_embedding_8/Shape?
:model_8/token_and_position_embedding_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2<
:model_8/token_and_position_embedding_8/strided_slice/stack?
<model_8/token_and_position_embedding_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_8/token_and_position_embedding_8/strided_slice/stack_1?
<model_8/token_and_position_embedding_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_8/token_and_position_embedding_8/strided_slice/stack_2?
4model_8/token_and_position_embedding_8/strided_sliceStridedSlice5model_8/token_and_position_embedding_8/Shape:output:0Cmodel_8/token_and_position_embedding_8/strided_slice/stack:output:0Emodel_8/token_and_position_embedding_8/strided_slice/stack_1:output:0Emodel_8/token_and_position_embedding_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_8/token_and_position_embedding_8/strided_slice?
2model_8/token_and_position_embedding_8/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_8/token_and_position_embedding_8/range/start?
2model_8/token_and_position_embedding_8/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_8/token_and_position_embedding_8/range/delta?
,model_8/token_and_position_embedding_8/rangeRange;model_8/token_and_position_embedding_8/range/start:output:0=model_8/token_and_position_embedding_8/strided_slice:output:0;model_8/token_and_position_embedding_8/range/delta:output:0*#
_output_shapes
:?????????2.
,model_8/token_and_position_embedding_8/range?
Dmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookupResourceGatherLmodel_8_token_and_position_embedding_8_embedding_17_embedding_lookup_48312305model_8/token_and_position_embedding_8/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*_
_classU
SQloc:@model_8/token_and_position_embedding_8/embedding_17/embedding_lookup/4831230*'
_output_shapes
:????????? *
dtype02F
Dmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup?
Mmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup/IdentityIdentityMmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*_
_classU
SQloc:@model_8/token_and_position_embedding_8/embedding_17/embedding_lookup/4831230*'
_output_shapes
:????????? 2O
Mmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup/Identity?
Omodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1IdentityVmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2Q
Omodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1?
8model_8/token_and_position_embedding_8/embedding_16/CastCastinput_9*

DstT0*

SrcT0*'
_output_shapes
:?????????(2:
8model_8/token_and_position_embedding_8/embedding_16/Cast?
Dmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookupResourceGatherLmodel_8_token_and_position_embedding_8_embedding_16_embedding_lookup_4831236<model_8/token_and_position_embedding_8/embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*_
_classU
SQloc:@model_8/token_and_position_embedding_8/embedding_16/embedding_lookup/4831236*+
_output_shapes
:?????????( *
dtype02F
Dmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup?
Mmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup/IdentityIdentityMmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*_
_classU
SQloc:@model_8/token_and_position_embedding_8/embedding_16/embedding_lookup/4831236*+
_output_shapes
:?????????( 2O
Mmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup/Identity?
Omodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1IdentityVmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2Q
Omodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1?
*model_8/token_and_position_embedding_8/addAddV2Xmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup/Identity_1:output:0Xmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2,
*model_8/token_and_position_embedding_8/add?
=model_8/transformer_block_8/multi_head_self_attention_8/ShapeShape.model_8/token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:2?
=model_8/transformer_block_8/multi_head_self_attention_8/Shape?
Kmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Kmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice/stack?
Mmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Mmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice/stack_1?
Mmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Mmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice/stack_2?
Emodel_8/transformer_block_8/multi_head_self_attention_8/strided_sliceStridedSliceFmodel_8/transformer_block_8/multi_head_self_attention_8/Shape:output:0Tmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice/stack:output:0Vmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice/stack_1:output:0Vmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2G
Emodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice?
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpReadVariableOpbmodel_8_transformer_block_8_multi_head_self_attention_8_dense_72_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02[
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axes?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/free?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ShapeShape.model_8/token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Shape?
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axis?
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2GatherV2Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/free:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2?
Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis?
Umodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1GatherV2Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axes:output:0cmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Umodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ProdProd\model_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_1?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod_1Prod^model_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1:output:0[model_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod_1?
Vmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat/axis?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concatConcatV2Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/free:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/axes:output:0_model_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/stackPackXmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod:output:0Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/stack?
Tmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/transpose	Transpose.model_8/token_and_position_embedding_8/add:z:0Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2V
Tmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/transpose?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReshapeReshapeXmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/transpose:y:0Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Reshape?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/MatMulMatMul[model_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Reshape:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/MatMul?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_2?
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1/axis?
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1ConcatV2\model_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0[model_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/Const_2:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1?
Jmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/TensordotReshape[model_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/MatMul:product:0\model_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2L
Jmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot?
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpReadVariableOp`model_8_transformer_block_8_multi_head_self_attention_8_dense_72_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?
Hmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/BiasAddBiasAddSmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot:output:0_model_8/transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2J
Hmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd?
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpReadVariableOpbmodel_8_transformer_block_8_multi_head_self_attention_8_dense_73_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02[
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axes?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/free?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ShapeShape.model_8/token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Shape?
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axis?
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2GatherV2Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/free:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2?
Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis?
Umodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1GatherV2Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axes:output:0cmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Umodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ProdProd\model_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_1?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod_1Prod^model_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1:output:0[model_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod_1?
Vmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat/axis?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concatConcatV2Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/free:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/axes:output:0_model_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/stackPackXmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod:output:0Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/stack?
Tmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/transpose	Transpose.model_8/token_and_position_embedding_8/add:z:0Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2V
Tmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/transpose?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReshapeReshapeXmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/transpose:y:0Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Reshape?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/MatMulMatMul[model_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Reshape:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/MatMul?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_2?
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1/axis?
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1ConcatV2\model_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0[model_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/Const_2:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1?
Jmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/TensordotReshape[model_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/MatMul:product:0\model_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2L
Jmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot?
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpReadVariableOp`model_8_transformer_block_8_multi_head_self_attention_8_dense_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?
Hmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/BiasAddBiasAddSmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot:output:0_model_8/transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2J
Hmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd?
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpReadVariableOpbmodel_8_transformer_block_8_multi_head_self_attention_8_dense_74_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02[
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axes?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/free?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ShapeShape.model_8/token_and_position_embedding_8/add:z:0*
T0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Shape?
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axis?
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2GatherV2Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/free:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2?
Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis?
Umodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1GatherV2Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axes:output:0cmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Umodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ProdProd\model_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_1?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod_1Prod^model_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1:output:0[model_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod_1?
Vmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat/axis?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concatConcatV2Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/free:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/axes:output:0_model_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/stackPackXmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod:output:0Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/stack?
Tmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/transpose	Transpose.model_8/token_and_position_embedding_8/add:z:0Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2V
Tmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/transpose?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReshapeReshapeXmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/transpose:y:0Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Reshape?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/MatMulMatMul[model_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Reshape:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/MatMul?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_2?
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1/axis?
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1ConcatV2\model_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0[model_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/Const_2:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1?
Jmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/TensordotReshape[model_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/MatMul:product:0\model_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2L
Jmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot?
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpReadVariableOp`model_8_transformer_block_8_multi_head_self_attention_8_dense_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?
Hmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/BiasAddBiasAddSmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot:output:0_model_8/transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2J
Hmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd?
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2I
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape/1?
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2I
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape/2?
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2I
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape/3?
Emodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shapePackNmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice:output:0Pmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape/1:output:0Pmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape/2:output:0Pmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2G
Emodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape?
?model_8/transformer_block_8/multi_head_self_attention_8/ReshapeReshapeQmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd:output:0Nmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2A
?model_8/transformer_block_8/multi_head_self_attention_8/Reshape?
Fmodel_8/transformer_block_8/multi_head_self_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2H
Fmodel_8/transformer_block_8/multi_head_self_attention_8/transpose/perm?
Amodel_8/transformer_block_8/multi_head_self_attention_8/transpose	TransposeHmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape:output:0Omodel_8/transformer_block_8/multi_head_self_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2C
Amodel_8/transformer_block_8/multi_head_self_attention_8/transpose?
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2K
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape/1?
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2K
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape/2?
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2K
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape/3?
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shapePackNmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice:output:0Rmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape/1:output:0Rmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape/2:output:0Rmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2I
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape?
Amodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1ReshapeQmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd:output:0Pmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2C
Amodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1?
Hmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2J
Hmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_1/perm?
Cmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_1	TransposeJmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_1:output:0Qmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2E
Cmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_1?
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2K
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape/1?
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2K
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape/2?
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2K
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape/3?
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shapePackNmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice:output:0Rmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape/1:output:0Rmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape/2:output:0Rmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2I
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape?
Amodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2ReshapeQmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd:output:0Pmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2C
Amodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2?
Hmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2J
Hmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_2/perm?
Cmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_2	TransposeJmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_2:output:0Qmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2E
Cmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_2?
>model_8/transformer_block_8/multi_head_self_attention_8/MatMulBatchMatMulV2Emodel_8/transformer_block_8/multi_head_self_attention_8/transpose:y:0Gmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2@
>model_8/transformer_block_8/multi_head_self_attention_8/MatMul?
?model_8/transformer_block_8/multi_head_self_attention_8/Shape_1ShapeGmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_1:y:0*
T0*
_output_shapes
:2A
?model_8/transformer_block_8/multi_head_self_attention_8/Shape_1?
Mmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2O
Mmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1/stack?
Omodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_1?
Omodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_2?
Gmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1StridedSliceHmodel_8/transformer_block_8/multi_head_self_attention_8/Shape_1:output:0Vmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1/stack:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_1:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1?
<model_8/transformer_block_8/multi_head_self_attention_8/CastCastPmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2>
<model_8/transformer_block_8/multi_head_self_attention_8/Cast?
<model_8/transformer_block_8/multi_head_self_attention_8/SqrtSqrt@model_8/transformer_block_8/multi_head_self_attention_8/Cast:y:0*
T0*
_output_shapes
: 2>
<model_8/transformer_block_8/multi_head_self_attention_8/Sqrt?
?model_8/transformer_block_8/multi_head_self_attention_8/truedivRealDivGmodel_8/transformer_block_8/multi_head_self_attention_8/MatMul:output:0@model_8/transformer_block_8/multi_head_self_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2A
?model_8/transformer_block_8/multi_head_self_attention_8/truediv?
?model_8/transformer_block_8/multi_head_self_attention_8/SoftmaxSoftmaxCmodel_8/transformer_block_8/multi_head_self_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2A
?model_8/transformer_block_8/multi_head_self_attention_8/Softmax?
@model_8/transformer_block_8/multi_head_self_attention_8/MatMul_1BatchMatMulV2Imodel_8/transformer_block_8/multi_head_self_attention_8/Softmax:softmax:0Gmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2B
@model_8/transformer_block_8/multi_head_self_attention_8/MatMul_1?
Hmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2J
Hmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_3/perm?
Cmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_3	TransposeImodel_8/transformer_block_8/multi_head_self_attention_8/MatMul_1:output:0Qmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2E
Cmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_3?
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2K
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3/shape/1?
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3/shape/2?
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3/shapePackNmodel_8/transformer_block_8/multi_head_self_attention_8/strided_slice:output:0Rmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3/shape/1:output:0Rmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2I
Gmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3/shape?
Amodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3ReshapeGmodel_8/transformer_block_8/multi_head_self_attention_8/transpose_3:y:0Pmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2C
Amodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3?
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpReadVariableOpbmodel_8_transformer_block_8_multi_head_self_attention_8_dense_75_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02[
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axes?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/free?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ShapeShapeJmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Shape?
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axis?
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2GatherV2Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/free:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2?
Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis?
Umodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1GatherV2Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axes:output:0cmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Umodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const?
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ProdProd\model_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Omodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_1?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod_1Prod^model_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1:output:0[model_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod_1?
Vmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat/axis?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concatConcatV2Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/free:output:0Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/axes:output:0_model_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat?
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/stackPackXmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod:output:0Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/stack?
Tmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/transpose	TransposeJmodel_8/transformer_block_8/multi_head_self_attention_8/Reshape_3:output:0Zmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2V
Tmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/transpose?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReshapeReshapeXmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/transpose:y:0Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Reshape?
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/MatMulMatMul[model_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Reshape:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2S
Qmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/MatMul?
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_2?
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1/axis?
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1ConcatV2\model_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0[model_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/Const_2:output:0amodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Smodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1?
Jmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/TensordotReshape[model_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/MatMul:product:0\model_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 2L
Jmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot?
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpReadVariableOp`model_8_transformer_block_8_multi_head_self_attention_8_dense_75_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02Y
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?
Hmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/BiasAddBiasAddSmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot:output:0_model_8/transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2J
Hmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd?
/model_8/transformer_block_8/dropout_16/IdentityIdentityQmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 21
/model_8/transformer_block_8/dropout_16/Identity?
model_8/transformer_block_8/addAddV2.model_8/token_and_position_embedding_8/add:z:08model_8/transformer_block_8/dropout_16/Identity:output:0*
T0*+
_output_shapes
:?????????( 2!
model_8/transformer_block_8/add?
Qmodel_8/transformer_block_8/layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_8/transformer_block_8/layer_normalization_16/moments/mean/reduction_indices?
?model_8/transformer_block_8/layer_normalization_16/moments/meanMean#model_8/transformer_block_8/add:z:0Zmodel_8/transformer_block_8/layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2A
?model_8/transformer_block_8/layer_normalization_16/moments/mean?
Gmodel_8/transformer_block_8/layer_normalization_16/moments/StopGradientStopGradientHmodel_8/transformer_block_8/layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2I
Gmodel_8/transformer_block_8/layer_normalization_16/moments/StopGradient?
Lmodel_8/transformer_block_8/layer_normalization_16/moments/SquaredDifferenceSquaredDifference#model_8/transformer_block_8/add:z:0Pmodel_8/transformer_block_8/layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2N
Lmodel_8/transformer_block_8/layer_normalization_16/moments/SquaredDifference?
Umodel_8/transformer_block_8/layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_8/transformer_block_8/layer_normalization_16/moments/variance/reduction_indices?
Cmodel_8/transformer_block_8/layer_normalization_16/moments/varianceMeanPmodel_8/transformer_block_8/layer_normalization_16/moments/SquaredDifference:z:0^model_8/transformer_block_8/layer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2E
Cmodel_8/transformer_block_8/layer_normalization_16/moments/variance?
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52D
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add/y?
@model_8/transformer_block_8/layer_normalization_16/batchnorm/addAddV2Lmodel_8/transformer_block_8/layer_normalization_16/moments/variance:output:0Kmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2B
@model_8/transformer_block_8/layer_normalization_16/batchnorm/add?
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/RsqrtRsqrtDmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2D
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/Rsqrt?
Omodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_8_transformer_block_8_layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp?
@model_8/transformer_block_8/layer_normalization_16/batchnorm/mulMulFmodel_8/transformer_block_8/layer_normalization_16/batchnorm/Rsqrt:y:0Wmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@model_8/transformer_block_8/layer_normalization_16/batchnorm/mul?
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul_1Mul#model_8/transformer_block_8/add:z:0Dmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2D
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul_1?
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul_2MulHmodel_8/transformer_block_8/layer_normalization_16/moments/mean:output:0Dmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2D
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul_2?
Kmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOpTmodel_8_transformer_block_8_layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp?
@model_8/transformer_block_8/layer_normalization_16/batchnorm/subSubSmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp:value:0Fmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2B
@model_8/transformer_block_8/layer_normalization_16/batchnorm/sub?
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add_1AddV2Fmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul_1:z:0Dmodel_8/transformer_block_8/layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2D
Bmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add_1?
Jmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOpReadVariableOpSmodel_8_transformer_block_8_sequential_8_dense_76_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02L
Jmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp?
@model_8/transformer_block_8/sequential_8/dense_76/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_8/transformer_block_8/sequential_8/dense_76/Tensordot/axes?
@model_8/transformer_block_8/sequential_8/dense_76/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_8/transformer_block_8/sequential_8/dense_76/Tensordot/free?
Amodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/ShapeShapeFmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:2C
Amodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Shape?
Imodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2/axis?
Dmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2GatherV2Jmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Shape:output:0Imodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/free:output:0Rmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2?
Kmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1/axis?
Fmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1GatherV2Jmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Shape:output:0Imodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/axes:output:0Tmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1?
Amodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Const?
@model_8/transformer_block_8/sequential_8/dense_76/Tensordot/ProdProdMmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2:output:0Jmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_8/transformer_block_8/sequential_8/dense_76/Tensordot/Prod?
Cmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Const_1?
Bmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Prod_1ProdOmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2_1:output:0Lmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Prod_1?
Gmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat/axis?
Bmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concatConcatV2Imodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/free:output:0Imodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/axes:output:0Pmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat?
Amodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/stackPackImodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Prod:output:0Kmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/stack?
Emodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/transpose	TransposeFmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add_1:z:0Kmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2G
Emodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/transpose?
Cmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/ReshapeReshapeImodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/transpose:y:0Jmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2E
Cmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Reshape?
Bmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/MatMulMatMulLmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Reshape:output:0Rmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2D
Bmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/MatMul?
Cmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Const_2?
Imodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat_1/axis?
Dmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat_1ConcatV2Mmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/GatherV2:output:0Lmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/Const_2:output:0Rmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat_1?
;model_8/transformer_block_8/sequential_8/dense_76/TensordotReshapeLmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/MatMul:product:0Mmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2=
;model_8/transformer_block_8/sequential_8/dense_76/Tensordot?
Hmodel_8/transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOpReadVariableOpQmodel_8_transformer_block_8_sequential_8_dense_76_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_8/transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp?
9model_8/transformer_block_8/sequential_8/dense_76/BiasAddBiasAddDmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot:output:0Pmodel_8/transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2;
9model_8/transformer_block_8/sequential_8/dense_76/BiasAdd?
6model_8/transformer_block_8/sequential_8/dense_76/ReluReluBmodel_8/transformer_block_8/sequential_8/dense_76/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 28
6model_8/transformer_block_8/sequential_8/dense_76/Relu?
Jmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOpReadVariableOpSmodel_8_transformer_block_8_sequential_8_dense_77_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02L
Jmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp?
@model_8/transformer_block_8/sequential_8/dense_77/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@model_8/transformer_block_8/sequential_8/dense_77/Tensordot/axes?
@model_8/transformer_block_8/sequential_8/dense_77/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@model_8/transformer_block_8/sequential_8/dense_77/Tensordot/free?
Amodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/ShapeShapeDmodel_8/transformer_block_8/sequential_8/dense_76/Relu:activations:0*
T0*
_output_shapes
:2C
Amodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Shape?
Imodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2/axis?
Dmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2GatherV2Jmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Shape:output:0Imodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/free:output:0Rmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2?
Kmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1/axis?
Fmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1GatherV2Jmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Shape:output:0Imodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/axes:output:0Tmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Fmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1?
Amodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Amodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Const?
@model_8/transformer_block_8/sequential_8/dense_77/Tensordot/ProdProdMmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2:output:0Jmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@model_8/transformer_block_8/sequential_8/dense_77/Tensordot/Prod?
Cmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Const_1?
Bmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Prod_1ProdOmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2_1:output:0Lmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Prod_1?
Gmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat/axis?
Bmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concatConcatV2Imodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/free:output:0Imodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/axes:output:0Pmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat?
Amodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/stackPackImodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Prod:output:0Kmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Amodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/stack?
Emodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/transpose	TransposeDmodel_8/transformer_block_8/sequential_8/dense_76/Relu:activations:0Kmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2G
Emodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/transpose?
Cmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/ReshapeReshapeImodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/transpose:y:0Jmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2E
Cmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Reshape?
Bmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/MatMulMatMulLmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Reshape:output:0Rmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2D
Bmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/MatMul?
Cmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Const_2?
Imodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Imodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat_1/axis?
Dmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat_1ConcatV2Mmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/GatherV2:output:0Lmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/Const_2:output:0Rmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat_1?
;model_8/transformer_block_8/sequential_8/dense_77/TensordotReshapeLmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/MatMul:product:0Mmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2=
;model_8/transformer_block_8/sequential_8/dense_77/Tensordot?
Hmodel_8/transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOpReadVariableOpQmodel_8_transformer_block_8_sequential_8_dense_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02J
Hmodel_8/transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp?
9model_8/transformer_block_8/sequential_8/dense_77/BiasAddBiasAddDmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot:output:0Pmodel_8/transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2;
9model_8/transformer_block_8/sequential_8/dense_77/BiasAdd?
/model_8/transformer_block_8/dropout_17/IdentityIdentityBmodel_8/transformer_block_8/sequential_8/dense_77/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 21
/model_8/transformer_block_8/dropout_17/Identity?
!model_8/transformer_block_8/add_1AddV2Fmodel_8/transformer_block_8/layer_normalization_16/batchnorm/add_1:z:08model_8/transformer_block_8/dropout_17/Identity:output:0*
T0*+
_output_shapes
:?????????( 2#
!model_8/transformer_block_8/add_1?
Qmodel_8/transformer_block_8/layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_8/transformer_block_8/layer_normalization_17/moments/mean/reduction_indices?
?model_8/transformer_block_8/layer_normalization_17/moments/meanMean%model_8/transformer_block_8/add_1:z:0Zmodel_8/transformer_block_8/layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2A
?model_8/transformer_block_8/layer_normalization_17/moments/mean?
Gmodel_8/transformer_block_8/layer_normalization_17/moments/StopGradientStopGradientHmodel_8/transformer_block_8/layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2I
Gmodel_8/transformer_block_8/layer_normalization_17/moments/StopGradient?
Lmodel_8/transformer_block_8/layer_normalization_17/moments/SquaredDifferenceSquaredDifference%model_8/transformer_block_8/add_1:z:0Pmodel_8/transformer_block_8/layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 2N
Lmodel_8/transformer_block_8/layer_normalization_17/moments/SquaredDifference?
Umodel_8/transformer_block_8/layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_8/transformer_block_8/layer_normalization_17/moments/variance/reduction_indices?
Cmodel_8/transformer_block_8/layer_normalization_17/moments/varianceMeanPmodel_8/transformer_block_8/layer_normalization_17/moments/SquaredDifference:z:0^model_8/transformer_block_8/layer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2E
Cmodel_8/transformer_block_8/layer_normalization_17/moments/variance?
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52D
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add/y?
@model_8/transformer_block_8/layer_normalization_17/batchnorm/addAddV2Lmodel_8/transformer_block_8/layer_normalization_17/moments/variance:output:0Kmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2B
@model_8/transformer_block_8/layer_normalization_17/batchnorm/add?
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/RsqrtRsqrtDmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2D
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/Rsqrt?
Omodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_8_transformer_block_8_layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02Q
Omodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp?
@model_8/transformer_block_8/layer_normalization_17/batchnorm/mulMulFmodel_8/transformer_block_8/layer_normalization_17/batchnorm/Rsqrt:y:0Wmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2B
@model_8/transformer_block_8/layer_normalization_17/batchnorm/mul?
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul_1Mul%model_8/transformer_block_8/add_1:z:0Dmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2D
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul_1?
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul_2MulHmodel_8/transformer_block_8/layer_normalization_17/moments/mean:output:0Dmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2D
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul_2?
Kmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOpTmodel_8_transformer_block_8_layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02M
Kmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp?
@model_8/transformer_block_8/layer_normalization_17/batchnorm/subSubSmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp:value:0Fmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2B
@model_8/transformer_block_8/layer_normalization_17/batchnorm/sub?
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add_1AddV2Fmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul_1:z:0Dmodel_8/transformer_block_8/layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2D
Bmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add_1?
9model_8/global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9model_8/global_average_pooling1d_8/Mean/reduction_indices?
'model_8/global_average_pooling1d_8/MeanMeanFmodel_8/transformer_block_8/layer_normalization_17/batchnorm/add_1:z:0Bmodel_8/global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2)
'model_8/global_average_pooling1d_8/Mean?
(model_8/aux_output/MatMul/ReadVariableOpReadVariableOp1model_8_aux_output_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(model_8/aux_output/MatMul/ReadVariableOp?
model_8/aux_output/MatMulMatMul0model_8/global_average_pooling1d_8/Mean:output:00model_8/aux_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_8/aux_output/MatMul?
)model_8/aux_output/BiasAdd/ReadVariableOpReadVariableOp2model_8_aux_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_8/aux_output/BiasAdd/ReadVariableOp?
model_8/aux_output/BiasAddBiasAdd#model_8/aux_output/MatMul:product:01model_8/aux_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_8/aux_output/BiasAdd?
model_8/aux_output/SigmoidSigmoid#model_8/aux_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_8/aux_output/Sigmoid?
!model_8/concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_8/concatenate_8/concat/axis?
model_8/concatenate_8/concatConcatV2model_8/aux_output/Sigmoid:y:0	aux_input*model_8/concatenate_8/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
model_8/concatenate_8/concat?
&model_8/dense_78/MatMul/ReadVariableOpReadVariableOp/model_8_dense_78_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&model_8/dense_78/MatMul/ReadVariableOp?
model_8/dense_78/MatMulMatMul%model_8/concatenate_8/concat:output:0.model_8/dense_78/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_8/dense_78/MatMul?
'model_8/dense_78/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_78_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_8/dense_78/BiasAdd/ReadVariableOp?
model_8/dense_78/BiasAddBiasAdd!model_8/dense_78/MatMul:product:0/model_8/dense_78/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_8/dense_78/BiasAdd?
model_8/dense_78/ReluRelu!model_8/dense_78/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_8/dense_78/Relu?
&model_8/dense_79/MatMul/ReadVariableOpReadVariableOp/model_8_dense_79_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_8/dense_79/MatMul/ReadVariableOp?
model_8/dense_79/MatMulMatMul#model_8/dense_78/Relu:activations:0.model_8/dense_79/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_8/dense_79/MatMul?
'model_8/dense_79/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_79_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_8/dense_79/BiasAdd/ReadVariableOp?
model_8/dense_79/BiasAddBiasAdd!model_8/dense_79/MatMul:product:0/model_8/dense_79/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_8/dense_79/BiasAdd?
model_8/dense_79/ReluRelu!model_8/dense_79/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_8/dense_79/Relu?
&model_8/dense_80/MatMul/ReadVariableOpReadVariableOp/model_8_dense_80_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&model_8/dense_80/MatMul/ReadVariableOp?
model_8/dense_80/MatMulMatMul#model_8/dense_79/Relu:activations:0.model_8/dense_80/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_8/dense_80/MatMul?
'model_8/dense_80/BiasAdd/ReadVariableOpReadVariableOp0model_8_dense_80_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_8/dense_80/BiasAdd/ReadVariableOp?
model_8/dense_80/BiasAddBiasAdd!model_8/dense_80/MatMul:product:0/model_8/dense_80/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model_8/dense_80/BiasAdd?
model_8/dense_80/ReluRelu!model_8/dense_80/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model_8/dense_80/Relu?
)model_8/main_output/MatMul/ReadVariableOpReadVariableOp2model_8_main_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02+
)model_8/main_output/MatMul/ReadVariableOp?
model_8/main_output/MatMulMatMul#model_8/dense_80/Relu:activations:01model_8/main_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_8/main_output/MatMul?
*model_8/main_output/BiasAdd/ReadVariableOpReadVariableOp3model_8_main_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model_8/main_output/BiasAdd/ReadVariableOp?
model_8/main_output/BiasAddBiasAdd$model_8/main_output/MatMul:product:02model_8/main_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_8/main_output/BiasAdd?
model_8/main_output/SigmoidSigmoid$model_8/main_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_8/main_output/Sigmoidy
IdentityIdentitymodel_8/aux_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity~

Identity_1Identitymodel_8/main_output/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity_1?
NoOpNoOp*^model_8/aux_output/BiasAdd/ReadVariableOp)^model_8/aux_output/MatMul/ReadVariableOp(^model_8/dense_78/BiasAdd/ReadVariableOp'^model_8/dense_78/MatMul/ReadVariableOp(^model_8/dense_79/BiasAdd/ReadVariableOp'^model_8/dense_79/MatMul/ReadVariableOp(^model_8/dense_80/BiasAdd/ReadVariableOp'^model_8/dense_80/MatMul/ReadVariableOp+^model_8/main_output/BiasAdd/ReadVariableOp*^model_8/main_output/MatMul/ReadVariableOpE^model_8/token_and_position_embedding_8/embedding_16/embedding_lookupE^model_8/token_and_position_embedding_8/embedding_17/embedding_lookupL^model_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpP^model_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpL^model_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpP^model_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpX^model_8/transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpZ^model_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpX^model_8/transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpZ^model_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpX^model_8/transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpZ^model_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpX^model_8/transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpZ^model_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpI^model_8/transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOpK^model_8/transformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOpI^model_8/transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOpK^model_8/transformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)model_8/aux_output/BiasAdd/ReadVariableOp)model_8/aux_output/BiasAdd/ReadVariableOp2T
(model_8/aux_output/MatMul/ReadVariableOp(model_8/aux_output/MatMul/ReadVariableOp2R
'model_8/dense_78/BiasAdd/ReadVariableOp'model_8/dense_78/BiasAdd/ReadVariableOp2P
&model_8/dense_78/MatMul/ReadVariableOp&model_8/dense_78/MatMul/ReadVariableOp2R
'model_8/dense_79/BiasAdd/ReadVariableOp'model_8/dense_79/BiasAdd/ReadVariableOp2P
&model_8/dense_79/MatMul/ReadVariableOp&model_8/dense_79/MatMul/ReadVariableOp2R
'model_8/dense_80/BiasAdd/ReadVariableOp'model_8/dense_80/BiasAdd/ReadVariableOp2P
&model_8/dense_80/MatMul/ReadVariableOp&model_8/dense_80/MatMul/ReadVariableOp2X
*model_8/main_output/BiasAdd/ReadVariableOp*model_8/main_output/BiasAdd/ReadVariableOp2V
)model_8/main_output/MatMul/ReadVariableOp)model_8/main_output/MatMul/ReadVariableOp2?
Dmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookupDmodel_8/token_and_position_embedding_8/embedding_16/embedding_lookup2?
Dmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookupDmodel_8/token_and_position_embedding_8/embedding_17/embedding_lookup2?
Kmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOpKmodel_8/transformer_block_8/layer_normalization_16/batchnorm/ReadVariableOp2?
Omodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOpOmodel_8/transformer_block_8/layer_normalization_16/batchnorm/mul/ReadVariableOp2?
Kmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOpKmodel_8/transformer_block_8/layer_normalization_17/batchnorm/ReadVariableOp2?
Omodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOpOmodel_8/transformer_block_8/layer_normalization_17/batchnorm/mul/ReadVariableOp2?
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpWmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp2?
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpYmodel_8/transformer_block_8/multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp2?
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpWmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp2?
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpYmodel_8/transformer_block_8/multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp2?
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpWmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp2?
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpYmodel_8/transformer_block_8/multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp2?
Wmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpWmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp2?
Ymodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpYmodel_8/transformer_block_8/multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp2?
Hmodel_8/transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOpHmodel_8/transformer_block_8/sequential_8/dense_76/BiasAdd/ReadVariableOp2?
Jmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOpJmodel_8/transformer_block_8/sequential_8/dense_76/Tensordot/ReadVariableOp2?
Hmodel_8/transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOpHmodel_8/transformer_block_8/sequential_8/dense_77/BiasAdd/ReadVariableOp2?
Jmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOpJmodel_8/transformer_block_8/sequential_8/dense_77/Tensordot/ReadVariableOp:P L
'
_output_shapes
:?????????(
!
_user_specified_name	input_9:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input
?
?
-__inference_main_output_layer_call_fn_4834588

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
H__inference_main_output_layer_call_and_return_conditional_losses_48321522
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
?
?
)__inference_model_8_layer_call_fn_4832889
input_9
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
StatefulPartitionedCallStatefulPartitionedCallinput_9	aux_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
D__inference_model_8_layer_call_and_return_conditional_losses_48327642
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
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????(
!
_user_specified_name	input_9:RN
'
_output_shapes
:?????????
#
_user_specified_name	aux_input
?
?
%__inference_signature_wrapper_4833103
	aux_input
input_9
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
StatefulPartitionedCallStatefulPartitionedCallinput_9	aux_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_48315232
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
^:?????????:?????????(: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	aux_input:PL
'
_output_shapes
:?????????(
!
_user_specified_name	input_9
?
s
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_4834480

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
*__inference_dense_79_layer_call_fn_4834548

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
E__inference_dense_79_layer_call_and_return_conditional_losses_48321182
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
?!
?
E__inference_dense_76_layer_call_and_return_conditional_losses_4834779

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
?
?
)__inference_model_8_layer_call_fn_4833231
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
D__inference_model_8_layer_call_and_return_conditional_losses_48327642
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
^:?????????(:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????(
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
??
?I
#__inference__traced_restore_4835447
file_prefix4
"assignvariableop_aux_output_kernel: 0
"assignvariableop_1_aux_output_bias:4
"assignvariableop_2_dense_78_kernel:@.
 assignvariableop_3_dense_78_bias:@4
"assignvariableop_4_dense_79_kernel:@@.
 assignvariableop_5_dense_79_bias:@4
"assignvariableop_6_dense_80_kernel:@@.
 assignvariableop_7_dense_80_bias:@7
%assignvariableop_8_main_output_kernel:@1
#assignvariableop_9_main_output_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: \
Jassignvariableop_15_token_and_position_embedding_8_embedding_16_embeddings: \
Jassignvariableop_16_token_and_position_embedding_8_embedding_17_embeddings:( e
Sassignvariableop_17_transformer_block_8_multi_head_self_attention_8_dense_72_kernel:  _
Qassignvariableop_18_transformer_block_8_multi_head_self_attention_8_dense_72_bias: e
Sassignvariableop_19_transformer_block_8_multi_head_self_attention_8_dense_73_kernel:  _
Qassignvariableop_20_transformer_block_8_multi_head_self_attention_8_dense_73_bias: e
Sassignvariableop_21_transformer_block_8_multi_head_self_attention_8_dense_74_kernel:  _
Qassignvariableop_22_transformer_block_8_multi_head_self_attention_8_dense_74_bias: e
Sassignvariableop_23_transformer_block_8_multi_head_self_attention_8_dense_75_kernel:  _
Qassignvariableop_24_transformer_block_8_multi_head_self_attention_8_dense_75_bias: 5
#assignvariableop_25_dense_76_kernel:  /
!assignvariableop_26_dense_76_bias: 5
#assignvariableop_27_dense_77_kernel:  /
!assignvariableop_28_dense_77_bias: R
Dassignvariableop_29_transformer_block_8_layer_normalization_16_gamma: Q
Cassignvariableop_30_transformer_block_8_layer_normalization_16_beta: R
Dassignvariableop_31_transformer_block_8_layer_normalization_17_gamma: Q
Cassignvariableop_32_transformer_block_8_layer_normalization_17_beta: #
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
*assignvariableop_45_adam_dense_78_kernel_m:@6
(assignvariableop_46_adam_dense_78_bias_m:@<
*assignvariableop_47_adam_dense_79_kernel_m:@@6
(assignvariableop_48_adam_dense_79_bias_m:@<
*assignvariableop_49_adam_dense_80_kernel_m:@@6
(assignvariableop_50_adam_dense_80_bias_m:@?
-assignvariableop_51_adam_main_output_kernel_m:@9
+assignvariableop_52_adam_main_output_bias_m:c
Qassignvariableop_53_adam_token_and_position_embedding_8_embedding_16_embeddings_m: c
Qassignvariableop_54_adam_token_and_position_embedding_8_embedding_17_embeddings_m:( l
Zassignvariableop_55_adam_transformer_block_8_multi_head_self_attention_8_dense_72_kernel_m:  f
Xassignvariableop_56_adam_transformer_block_8_multi_head_self_attention_8_dense_72_bias_m: l
Zassignvariableop_57_adam_transformer_block_8_multi_head_self_attention_8_dense_73_kernel_m:  f
Xassignvariableop_58_adam_transformer_block_8_multi_head_self_attention_8_dense_73_bias_m: l
Zassignvariableop_59_adam_transformer_block_8_multi_head_self_attention_8_dense_74_kernel_m:  f
Xassignvariableop_60_adam_transformer_block_8_multi_head_self_attention_8_dense_74_bias_m: l
Zassignvariableop_61_adam_transformer_block_8_multi_head_self_attention_8_dense_75_kernel_m:  f
Xassignvariableop_62_adam_transformer_block_8_multi_head_self_attention_8_dense_75_bias_m: <
*assignvariableop_63_adam_dense_76_kernel_m:  6
(assignvariableop_64_adam_dense_76_bias_m: <
*assignvariableop_65_adam_dense_77_kernel_m:  6
(assignvariableop_66_adam_dense_77_bias_m: Y
Kassignvariableop_67_adam_transformer_block_8_layer_normalization_16_gamma_m: X
Jassignvariableop_68_adam_transformer_block_8_layer_normalization_16_beta_m: Y
Kassignvariableop_69_adam_transformer_block_8_layer_normalization_17_gamma_m: X
Jassignvariableop_70_adam_transformer_block_8_layer_normalization_17_beta_m: >
,assignvariableop_71_adam_aux_output_kernel_v: 8
*assignvariableop_72_adam_aux_output_bias_v:<
*assignvariableop_73_adam_dense_78_kernel_v:@6
(assignvariableop_74_adam_dense_78_bias_v:@<
*assignvariableop_75_adam_dense_79_kernel_v:@@6
(assignvariableop_76_adam_dense_79_bias_v:@<
*assignvariableop_77_adam_dense_80_kernel_v:@@6
(assignvariableop_78_adam_dense_80_bias_v:@?
-assignvariableop_79_adam_main_output_kernel_v:@9
+assignvariableop_80_adam_main_output_bias_v:c
Qassignvariableop_81_adam_token_and_position_embedding_8_embedding_16_embeddings_v: c
Qassignvariableop_82_adam_token_and_position_embedding_8_embedding_17_embeddings_v:( l
Zassignvariableop_83_adam_transformer_block_8_multi_head_self_attention_8_dense_72_kernel_v:  f
Xassignvariableop_84_adam_transformer_block_8_multi_head_self_attention_8_dense_72_bias_v: l
Zassignvariableop_85_adam_transformer_block_8_multi_head_self_attention_8_dense_73_kernel_v:  f
Xassignvariableop_86_adam_transformer_block_8_multi_head_self_attention_8_dense_73_bias_v: l
Zassignvariableop_87_adam_transformer_block_8_multi_head_self_attention_8_dense_74_kernel_v:  f
Xassignvariableop_88_adam_transformer_block_8_multi_head_self_attention_8_dense_74_bias_v: l
Zassignvariableop_89_adam_transformer_block_8_multi_head_self_attention_8_dense_75_kernel_v:  f
Xassignvariableop_90_adam_transformer_block_8_multi_head_self_attention_8_dense_75_bias_v: <
*assignvariableop_91_adam_dense_76_kernel_v:  6
(assignvariableop_92_adam_dense_76_bias_v: <
*assignvariableop_93_adam_dense_77_kernel_v:  6
(assignvariableop_94_adam_dense_77_bias_v: Y
Kassignvariableop_95_adam_transformer_block_8_layer_normalization_16_gamma_v: X
Jassignvariableop_96_adam_transformer_block_8_layer_normalization_16_beta_v: Y
Kassignvariableop_97_adam_transformer_block_8_layer_normalization_17_gamma_v: X
Jassignvariableop_98_adam_transformer_block_8_layer_normalization_17_beta_v: 
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
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_78_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_78_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_79_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_79_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_80_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_80_biasIdentity_7:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOpJassignvariableop_15_token_and_position_embedding_8_embedding_16_embeddingsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpJassignvariableop_16_token_and_position_embedding_8_embedding_17_embeddingsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpSassignvariableop_17_transformer_block_8_multi_head_self_attention_8_dense_72_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpQassignvariableop_18_transformer_block_8_multi_head_self_attention_8_dense_72_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpSassignvariableop_19_transformer_block_8_multi_head_self_attention_8_dense_73_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpQassignvariableop_20_transformer_block_8_multi_head_self_attention_8_dense_73_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpSassignvariableop_21_transformer_block_8_multi_head_self_attention_8_dense_74_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpQassignvariableop_22_transformer_block_8_multi_head_self_attention_8_dense_74_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpSassignvariableop_23_transformer_block_8_multi_head_self_attention_8_dense_75_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpQassignvariableop_24_transformer_block_8_multi_head_self_attention_8_dense_75_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp#assignvariableop_25_dense_76_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp!assignvariableop_26_dense_76_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp#assignvariableop_27_dense_77_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp!assignvariableop_28_dense_77_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpDassignvariableop_29_transformer_block_8_layer_normalization_16_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpCassignvariableop_30_transformer_block_8_layer_normalization_16_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpDassignvariableop_31_transformer_block_8_layer_normalization_17_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpCassignvariableop_32_transformer_block_8_layer_normalization_17_betaIdentity_32:output:0"/device:CPU:0*
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
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_dense_78_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_dense_78_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_dense_79_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_dense_79_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_80_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_80_bias_mIdentity_50:output:0"/device:CPU:0*
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
AssignVariableOp_53AssignVariableOpQassignvariableop_53_adam_token_and_position_embedding_8_embedding_16_embeddings_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpQassignvariableop_54_adam_token_and_position_embedding_8_embedding_17_embeddings_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpZassignvariableop_55_adam_transformer_block_8_multi_head_self_attention_8_dense_72_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpXassignvariableop_56_adam_transformer_block_8_multi_head_self_attention_8_dense_72_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpZassignvariableop_57_adam_transformer_block_8_multi_head_self_attention_8_dense_73_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpXassignvariableop_58_adam_transformer_block_8_multi_head_self_attention_8_dense_73_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpZassignvariableop_59_adam_transformer_block_8_multi_head_self_attention_8_dense_74_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpXassignvariableop_60_adam_transformer_block_8_multi_head_self_attention_8_dense_74_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpZassignvariableop_61_adam_transformer_block_8_multi_head_self_attention_8_dense_75_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpXassignvariableop_62_adam_transformer_block_8_multi_head_self_attention_8_dense_75_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_dense_76_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_dense_76_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_dense_77_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_dense_77_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOpKassignvariableop_67_adam_transformer_block_8_layer_normalization_16_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOpJassignvariableop_68_adam_transformer_block_8_layer_normalization_16_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOpKassignvariableop_69_adam_transformer_block_8_layer_normalization_17_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOpJassignvariableop_70_adam_transformer_block_8_layer_normalization_17_beta_mIdentity_70:output:0"/device:CPU:0*
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
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_78_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_78_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_dense_79_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_dense_79_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_dense_80_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_dense_80_bias_vIdentity_78:output:0"/device:CPU:0*
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
AssignVariableOp_81AssignVariableOpQassignvariableop_81_adam_token_and_position_embedding_8_embedding_16_embeddings_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOpQassignvariableop_82_adam_token_and_position_embedding_8_embedding_17_embeddings_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOpZassignvariableop_83_adam_transformer_block_8_multi_head_self_attention_8_dense_72_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOpXassignvariableop_84_adam_transformer_block_8_multi_head_self_attention_8_dense_72_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOpZassignvariableop_85_adam_transformer_block_8_multi_head_self_attention_8_dense_73_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOpXassignvariableop_86_adam_transformer_block_8_multi_head_self_attention_8_dense_73_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOpZassignvariableop_87_adam_transformer_block_8_multi_head_self_attention_8_dense_74_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOpXassignvariableop_88_adam_transformer_block_8_multi_head_self_attention_8_dense_74_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOpZassignvariableop_89_adam_transformer_block_8_multi_head_self_attention_8_dense_75_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOpXassignvariableop_90_adam_transformer_block_8_multi_head_self_attention_8_dense_75_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_dense_76_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_dense_76_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_dense_77_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_dense_77_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOpKassignvariableop_95_adam_transformer_block_8_layer_normalization_16_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOpJassignvariableop_96_adam_transformer_block_8_layer_normalization_16_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOpKassignvariableop_97_adam_transformer_block_8_layer_normalization_17_gamma_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOpJassignvariableop_98_adam_transformer_block_8_layer_normalization_17_beta_vIdentity_98:output:0"/device:CPU:0*
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
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_4831773
x7
%embedding_17_embedding_lookup_4831760:( 7
%embedding_16_embedding_lookup_4831766: 
identity??embedding_16/embedding_lookup?embedding_17/embedding_lookup?
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
embedding_17/embedding_lookupResourceGather%embedding_17_embedding_lookup_4831760range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_17/embedding_lookup/4831760*'
_output_shapes
:????????? *
dtype02
embedding_17/embedding_lookup?
&embedding_17/embedding_lookup/IdentityIdentity&embedding_17/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_17/embedding_lookup/4831760*'
_output_shapes
:????????? 2(
&embedding_17/embedding_lookup/Identity?
(embedding_17/embedding_lookup/Identity_1Identity/embedding_17/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2*
(embedding_17/embedding_lookup/Identity_1r
embedding_16/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:?????????(2
embedding_16/Cast?
embedding_16/embedding_lookupResourceGather%embedding_16_embedding_lookup_4831766embedding_16/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*8
_class.
,*loc:@embedding_16/embedding_lookup/4831766*+
_output_shapes
:?????????( *
dtype02
embedding_16/embedding_lookup?
&embedding_16/embedding_lookup/IdentityIdentity&embedding_16/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*8
_class.
,*loc:@embedding_16/embedding_lookup/4831766*+
_output_shapes
:?????????( 2(
&embedding_16/embedding_lookup/Identity?
(embedding_16/embedding_lookup/Identity_1Identity/embedding_16/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????( 2*
(embedding_16/embedding_lookup/Identity_1?
addAddV21embedding_16/embedding_lookup/Identity_1:output:01embedding_17/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????( 2
addf
IdentityIdentityadd:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp^embedding_16/embedding_lookup^embedding_17/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 2>
embedding_16/embedding_lookupembedding_16/embedding_lookup2>
embedding_17/embedding_lookupembedding_17/embedding_lookup:J F
'
_output_shapes
:?????????(

_user_specified_namex
??
?
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_4834206

inputsX
Fmulti_head_self_attention_8_dense_72_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_72_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_73_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_73_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_74_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_74_biasadd_readvariableop_resource: X
Fmulti_head_self_attention_8_dense_75_tensordot_readvariableop_resource:  R
Dmulti_head_self_attention_8_dense_75_biasadd_readvariableop_resource: J
<layer_normalization_16_batchnorm_mul_readvariableop_resource: F
8layer_normalization_16_batchnorm_readvariableop_resource: I
7sequential_8_dense_76_tensordot_readvariableop_resource:  C
5sequential_8_dense_76_biasadd_readvariableop_resource: I
7sequential_8_dense_77_tensordot_readvariableop_resource:  C
5sequential_8_dense_77_biasadd_readvariableop_resource: J
<layer_normalization_17_batchnorm_mul_readvariableop_resource: F
8layer_normalization_17_batchnorm_readvariableop_resource: 
identity??/layer_normalization_16/batchnorm/ReadVariableOp?3layer_normalization_16/batchnorm/mul/ReadVariableOp?/layer_normalization_17/batchnorm/ReadVariableOp?3layer_normalization_17/batchnorm/mul/ReadVariableOp?;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?,sequential_8/dense_76/BiasAdd/ReadVariableOp?.sequential_8/dense_76/Tensordot/ReadVariableOp?,sequential_8/dense_77/BiasAdd/ReadVariableOp?.sequential_8/dense_77/Tensordot/ReadVariableOp|
!multi_head_self_attention_8/ShapeShapeinputs*
T0*
_output_shapes
:2#
!multi_head_self_attention_8/Shape?
/multi_head_self_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention_8/strided_slice/stack?
1multi_head_self_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_8/strided_slice/stack_1?
1multi_head_self_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention_8/strided_slice/stack_2?
)multi_head_self_attention_8/strided_sliceStridedSlice*multi_head_self_attention_8/Shape:output:08multi_head_self_attention_8/strided_slice/stack:output:0:multi_head_self_attention_8/strided_slice/stack_1:output:0:multi_head_self_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention_8/strided_slice?
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_72_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_72/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_72/Tensordot/axes?
3multi_head_self_attention_8/dense_72/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_72/Tensordot/free?
4multi_head_self_attention_8/dense_72/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_72/Tensordot/Shape?
<multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_72/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_72/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_72/Tensordot/free:output:0Emulti_head_self_attention_8/dense_72/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_72/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_72/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_72/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_72/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_72/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_72/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_72/Tensordot/Const?
3multi_head_self_attention_8/dense_72/Tensordot/ProdProd@multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_72/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_72/Tensordot/Prod?
6multi_head_self_attention_8/dense_72/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_72/Tensordot/Const_1?
5multi_head_self_attention_8/dense_72/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_72/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_72/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_72/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_72/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_72/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_72/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_72/Tensordot/free:output:0<multi_head_self_attention_8/dense_72/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_72/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_72/Tensordot/concat?
4multi_head_self_attention_8/dense_72/Tensordot/stackPack<multi_head_self_attention_8/dense_72/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_72/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_72/Tensordot/stack?
8multi_head_self_attention_8/dense_72/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_72/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_72/Tensordot/transpose?
6multi_head_self_attention_8/dense_72/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_72/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_72/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_72/Tensordot/Reshape?
5multi_head_self_attention_8/dense_72/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_72/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_72/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_72/Tensordot/MatMul?
6multi_head_self_attention_8/dense_72/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_72/Tensordot/Const_2?
<multi_head_self_attention_8/dense_72/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_72/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_72/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_72/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_72/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_72/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_72/Tensordot/concat_1?
.multi_head_self_attention_8/dense_72/TensordotReshape?multi_head_self_attention_8/dense_72/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_72/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_72/Tensordot?
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_72_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_72/BiasAddBiasAdd7multi_head_self_attention_8/dense_72/Tensordot:output:0Cmulti_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_72/BiasAdd?
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_73_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_73/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_73/Tensordot/axes?
3multi_head_self_attention_8/dense_73/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_73/Tensordot/free?
4multi_head_self_attention_8/dense_73/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_73/Tensordot/Shape?
<multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_73/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_73/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_73/Tensordot/free:output:0Emulti_head_self_attention_8/dense_73/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_73/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_73/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_73/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_73/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_73/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_73/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_73/Tensordot/Const?
3multi_head_self_attention_8/dense_73/Tensordot/ProdProd@multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_73/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_73/Tensordot/Prod?
6multi_head_self_attention_8/dense_73/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_73/Tensordot/Const_1?
5multi_head_self_attention_8/dense_73/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_73/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_73/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_73/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_73/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_73/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_73/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_73/Tensordot/free:output:0<multi_head_self_attention_8/dense_73/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_73/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_73/Tensordot/concat?
4multi_head_self_attention_8/dense_73/Tensordot/stackPack<multi_head_self_attention_8/dense_73/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_73/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_73/Tensordot/stack?
8multi_head_self_attention_8/dense_73/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_73/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_73/Tensordot/transpose?
6multi_head_self_attention_8/dense_73/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_73/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_73/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_73/Tensordot/Reshape?
5multi_head_self_attention_8/dense_73/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_73/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_73/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_73/Tensordot/MatMul?
6multi_head_self_attention_8/dense_73/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_73/Tensordot/Const_2?
<multi_head_self_attention_8/dense_73/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_73/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_73/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_73/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_73/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_73/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_73/Tensordot/concat_1?
.multi_head_self_attention_8/dense_73/TensordotReshape?multi_head_self_attention_8/dense_73/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_73/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_73/Tensordot?
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_73_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_73/BiasAddBiasAdd7multi_head_self_attention_8/dense_73/Tensordot:output:0Cmulti_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_73/BiasAdd?
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_74_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_74/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_74/Tensordot/axes?
3multi_head_self_attention_8/dense_74/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_74/Tensordot/free?
4multi_head_self_attention_8/dense_74/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_74/Tensordot/Shape?
<multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_74/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_74/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_74/Tensordot/free:output:0Emulti_head_self_attention_8/dense_74/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_74/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_74/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_74/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_74/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_74/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_74/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_74/Tensordot/Const?
3multi_head_self_attention_8/dense_74/Tensordot/ProdProd@multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_74/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_74/Tensordot/Prod?
6multi_head_self_attention_8/dense_74/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_74/Tensordot/Const_1?
5multi_head_self_attention_8/dense_74/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_74/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_74/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_74/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_74/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_74/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_74/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_74/Tensordot/free:output:0<multi_head_self_attention_8/dense_74/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_74/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_74/Tensordot/concat?
4multi_head_self_attention_8/dense_74/Tensordot/stackPack<multi_head_self_attention_8/dense_74/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_74/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_74/Tensordot/stack?
8multi_head_self_attention_8/dense_74/Tensordot/transpose	Transposeinputs>multi_head_self_attention_8/dense_74/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2:
8multi_head_self_attention_8/dense_74/Tensordot/transpose?
6multi_head_self_attention_8/dense_74/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_74/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_74/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_74/Tensordot/Reshape?
5multi_head_self_attention_8/dense_74/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_74/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_74/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_74/Tensordot/MatMul?
6multi_head_self_attention_8/dense_74/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_74/Tensordot/Const_2?
<multi_head_self_attention_8/dense_74/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_74/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_74/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_74/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_74/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_74/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_74/Tensordot/concat_1?
.multi_head_self_attention_8/dense_74/TensordotReshape?multi_head_self_attention_8/dense_74/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_74/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 20
.multi_head_self_attention_8/dense_74/Tensordot?
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_74_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_74/BiasAddBiasAdd7multi_head_self_attention_8/dense_74/Tensordot:output:0Cmulti_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2.
,multi_head_self_attention_8/dense_74/BiasAdd?
+multi_head_self_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+multi_head_self_attention_8/Reshape/shape/1?
+multi_head_self_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_8/Reshape/shape/2?
+multi_head_self_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention_8/Reshape/shape/3?
)multi_head_self_attention_8/Reshape/shapePack2multi_head_self_attention_8/strided_slice:output:04multi_head_self_attention_8/Reshape/shape/1:output:04multi_head_self_attention_8/Reshape/shape/2:output:04multi_head_self_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention_8/Reshape/shape?
#multi_head_self_attention_8/ReshapeReshape5multi_head_self_attention_8/dense_72/BiasAdd:output:02multi_head_self_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2%
#multi_head_self_attention_8/Reshape?
*multi_head_self_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention_8/transpose/perm?
%multi_head_self_attention_8/transpose	Transpose,multi_head_self_attention_8/Reshape:output:03multi_head_self_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/transpose?
-multi_head_self_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_1/shape/1?
-multi_head_self_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_1/shape/2?
-multi_head_self_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_1/shape/3?
+multi_head_self_attention_8/Reshape_1/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_1/shape/1:output:06multi_head_self_attention_8/Reshape_1/shape/2:output:06multi_head_self_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_1/shape?
%multi_head_self_attention_8/Reshape_1Reshape5multi_head_self_attention_8/dense_73/BiasAdd:output:04multi_head_self_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/Reshape_1?
,multi_head_self_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_1/perm?
'multi_head_self_attention_8/transpose_1	Transpose.multi_head_self_attention_8/Reshape_1:output:05multi_head_self_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_1?
-multi_head_self_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_2/shape/1?
-multi_head_self_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_2/shape/2?
-multi_head_self_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-multi_head_self_attention_8/Reshape_2/shape/3?
+multi_head_self_attention_8/Reshape_2/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_2/shape/1:output:06multi_head_self_attention_8/Reshape_2/shape/2:output:06multi_head_self_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_2/shape?
%multi_head_self_attention_8/Reshape_2Reshape5multi_head_self_attention_8/dense_74/BiasAdd:output:04multi_head_self_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2'
%multi_head_self_attention_8/Reshape_2?
,multi_head_self_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_2/perm?
'multi_head_self_attention_8/transpose_2	Transpose.multi_head_self_attention_8/Reshape_2:output:05multi_head_self_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_2?
"multi_head_self_attention_8/MatMulBatchMatMulV2)multi_head_self_attention_8/transpose:y:0+multi_head_self_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2$
"multi_head_self_attention_8/MatMul?
#multi_head_self_attention_8/Shape_1Shape+multi_head_self_attention_8/transpose_1:y:0*
T0*
_output_shapes
:2%
#multi_head_self_attention_8/Shape_1?
1multi_head_self_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????23
1multi_head_self_attention_8/strided_slice_1/stack?
3multi_head_self_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention_8/strided_slice_1/stack_1?
3multi_head_self_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/strided_slice_1/stack_2?
+multi_head_self_attention_8/strided_slice_1StridedSlice,multi_head_self_attention_8/Shape_1:output:0:multi_head_self_attention_8/strided_slice_1/stack:output:0<multi_head_self_attention_8/strided_slice_1/stack_1:output:0<multi_head_self_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+multi_head_self_attention_8/strided_slice_1?
 multi_head_self_attention_8/CastCast4multi_head_self_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 multi_head_self_attention_8/Cast?
 multi_head_self_attention_8/SqrtSqrt$multi_head_self_attention_8/Cast:y:0*
T0*
_output_shapes
: 2"
 multi_head_self_attention_8/Sqrt?
#multi_head_self_attention_8/truedivRealDiv+multi_head_self_attention_8/MatMul:output:0$multi_head_self_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_8/truediv?
#multi_head_self_attention_8/SoftmaxSoftmax'multi_head_self_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2%
#multi_head_self_attention_8/Softmax?
$multi_head_self_attention_8/MatMul_1BatchMatMulV2-multi_head_self_attention_8/Softmax:softmax:0+multi_head_self_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"??????????????????2&
$multi_head_self_attention_8/MatMul_1?
,multi_head_self_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2.
,multi_head_self_attention_8/transpose_3/perm?
'multi_head_self_attention_8/transpose_3	Transpose-multi_head_self_attention_8/MatMul_1:output:05multi_head_self_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"??????????????????2)
'multi_head_self_attention_8/transpose_3?
-multi_head_self_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-multi_head_self_attention_8/Reshape_3/shape/1?
-multi_head_self_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2/
-multi_head_self_attention_8/Reshape_3/shape/2?
+multi_head_self_attention_8/Reshape_3/shapePack2multi_head_self_attention_8/strided_slice:output:06multi_head_self_attention_8/Reshape_3/shape/1:output:06multi_head_self_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+multi_head_self_attention_8/Reshape_3/shape?
%multi_head_self_attention_8/Reshape_3Reshape+multi_head_self_attention_8/transpose_3:y:04multi_head_self_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2'
%multi_head_self_attention_8/Reshape_3?
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOpReadVariableOpFmulti_head_self_attention_8_dense_75_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02?
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp?
3multi_head_self_attention_8/dense_75/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3multi_head_self_attention_8/dense_75/Tensordot/axes?
3multi_head_self_attention_8/dense_75/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3multi_head_self_attention_8/dense_75/Tensordot/free?
4multi_head_self_attention_8/dense_75/Tensordot/ShapeShape.multi_head_self_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_75/Tensordot/Shape?
<multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_75/Tensordot/GatherV2/axis?
7multi_head_self_attention_8/dense_75/Tensordot/GatherV2GatherV2=multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_75/Tensordot/free:output:0Emulti_head_self_attention_8/dense_75/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7multi_head_self_attention_8/dense_75/Tensordot/GatherV2?
>multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis?
9multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1GatherV2=multi_head_self_attention_8/dense_75/Tensordot/Shape:output:0<multi_head_self_attention_8/dense_75/Tensordot/axes:output:0Gmulti_head_self_attention_8/dense_75/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9multi_head_self_attention_8/dense_75/Tensordot/GatherV2_1?
4multi_head_self_attention_8/dense_75/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4multi_head_self_attention_8/dense_75/Tensordot/Const?
3multi_head_self_attention_8/dense_75/Tensordot/ProdProd@multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0=multi_head_self_attention_8/dense_75/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3multi_head_self_attention_8/dense_75/Tensordot/Prod?
6multi_head_self_attention_8/dense_75/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_75/Tensordot/Const_1?
5multi_head_self_attention_8/dense_75/Tensordot/Prod_1ProdBmulti_head_self_attention_8/dense_75/Tensordot/GatherV2_1:output:0?multi_head_self_attention_8/dense_75/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5multi_head_self_attention_8/dense_75/Tensordot/Prod_1?
:multi_head_self_attention_8/dense_75/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:multi_head_self_attention_8/dense_75/Tensordot/concat/axis?
5multi_head_self_attention_8/dense_75/Tensordot/concatConcatV2<multi_head_self_attention_8/dense_75/Tensordot/free:output:0<multi_head_self_attention_8/dense_75/Tensordot/axes:output:0Cmulti_head_self_attention_8/dense_75/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5multi_head_self_attention_8/dense_75/Tensordot/concat?
4multi_head_self_attention_8/dense_75/Tensordot/stackPack<multi_head_self_attention_8/dense_75/Tensordot/Prod:output:0>multi_head_self_attention_8/dense_75/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention_8/dense_75/Tensordot/stack?
8multi_head_self_attention_8/dense_75/Tensordot/transpose	Transpose.multi_head_self_attention_8/Reshape_3:output:0>multi_head_self_attention_8/dense_75/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :?????????????????? 2:
8multi_head_self_attention_8/dense_75/Tensordot/transpose?
6multi_head_self_attention_8/dense_75/Tensordot/ReshapeReshape<multi_head_self_attention_8/dense_75/Tensordot/transpose:y:0=multi_head_self_attention_8/dense_75/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6multi_head_self_attention_8/dense_75/Tensordot/Reshape?
5multi_head_self_attention_8/dense_75/Tensordot/MatMulMatMul?multi_head_self_attention_8/dense_75/Tensordot/Reshape:output:0Emulti_head_self_attention_8/dense_75/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5multi_head_self_attention_8/dense_75/Tensordot/MatMul?
6multi_head_self_attention_8/dense_75/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6multi_head_self_attention_8/dense_75/Tensordot/Const_2?
<multi_head_self_attention_8/dense_75/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<multi_head_self_attention_8/dense_75/Tensordot/concat_1/axis?
7multi_head_self_attention_8/dense_75/Tensordot/concat_1ConcatV2@multi_head_self_attention_8/dense_75/Tensordot/GatherV2:output:0?multi_head_self_attention_8/dense_75/Tensordot/Const_2:output:0Emulti_head_self_attention_8/dense_75/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7multi_head_self_attention_8/dense_75/Tensordot/concat_1?
.multi_head_self_attention_8/dense_75/TensordotReshape?multi_head_self_attention_8/dense_75/Tensordot/MatMul:product:0@multi_head_self_attention_8/dense_75/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :?????????????????? 20
.multi_head_self_attention_8/dense_75/Tensordot?
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOpReadVariableOpDmulti_head_self_attention_8_dense_75_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp?
,multi_head_self_attention_8/dense_75/BiasAddBiasAdd7multi_head_self_attention_8/dense_75/Tensordot:output:0Cmulti_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2.
,multi_head_self_attention_8/dense_75/BiasAdd?
dropout_16/IdentityIdentity5multi_head_self_attention_8/dense_75/BiasAdd:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
dropout_16/Identityo
addAddV2inputsdropout_16/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add?
5layer_normalization_16/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_16/moments/mean/reduction_indices?
#layer_normalization_16/moments/meanMeanadd:z:0>layer_normalization_16/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_16/moments/mean?
+layer_normalization_16/moments/StopGradientStopGradient,layer_normalization_16/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_16/moments/StopGradient?
0layer_normalization_16/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_16/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_16/moments/SquaredDifference?
9layer_normalization_16/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_16/moments/variance/reduction_indices?
'layer_normalization_16/moments/varianceMean4layer_normalization_16/moments/SquaredDifference:z:0Blayer_normalization_16/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_16/moments/variance?
&layer_normalization_16/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_16/batchnorm/add/y?
$layer_normalization_16/batchnorm/addAddV20layer_normalization_16/moments/variance:output:0/layer_normalization_16/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_16/batchnorm/add?
&layer_normalization_16/batchnorm/RsqrtRsqrt(layer_normalization_16/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_16/batchnorm/Rsqrt?
3layer_normalization_16/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_16_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_16/batchnorm/mul/ReadVariableOp?
$layer_normalization_16/batchnorm/mulMul*layer_normalization_16/batchnorm/Rsqrt:y:0;layer_normalization_16/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_16/batchnorm/mul?
&layer_normalization_16/batchnorm/mul_1Muladd:z:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/mul_1?
&layer_normalization_16/batchnorm/mul_2Mul,layer_normalization_16/moments/mean:output:0(layer_normalization_16/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/mul_2?
/layer_normalization_16/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_16_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_16/batchnorm/ReadVariableOp?
$layer_normalization_16/batchnorm/subSub7layer_normalization_16/batchnorm/ReadVariableOp:value:0*layer_normalization_16/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_16/batchnorm/sub?
&layer_normalization_16/batchnorm/add_1AddV2*layer_normalization_16/batchnorm/mul_1:z:0(layer_normalization_16/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_16/batchnorm/add_1?
.sequential_8/dense_76/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_76_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_8/dense_76/Tensordot/ReadVariableOp?
$sequential_8/dense_76/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_8/dense_76/Tensordot/axes?
$sequential_8/dense_76/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_8/dense_76/Tensordot/free?
%sequential_8/dense_76/Tensordot/ShapeShape*layer_normalization_16/batchnorm/add_1:z:0*
T0*
_output_shapes
:2'
%sequential_8/dense_76/Tensordot/Shape?
-sequential_8/dense_76/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_76/Tensordot/GatherV2/axis?
(sequential_8/dense_76/Tensordot/GatherV2GatherV2.sequential_8/dense_76/Tensordot/Shape:output:0-sequential_8/dense_76/Tensordot/free:output:06sequential_8/dense_76/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_8/dense_76/Tensordot/GatherV2?
/sequential_8/dense_76/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_8/dense_76/Tensordot/GatherV2_1/axis?
*sequential_8/dense_76/Tensordot/GatherV2_1GatherV2.sequential_8/dense_76/Tensordot/Shape:output:0-sequential_8/dense_76/Tensordot/axes:output:08sequential_8/dense_76/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_8/dense_76/Tensordot/GatherV2_1?
%sequential_8/dense_76/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_8/dense_76/Tensordot/Const?
$sequential_8/dense_76/Tensordot/ProdProd1sequential_8/dense_76/Tensordot/GatherV2:output:0.sequential_8/dense_76/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_8/dense_76/Tensordot/Prod?
'sequential_8/dense_76/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_76/Tensordot/Const_1?
&sequential_8/dense_76/Tensordot/Prod_1Prod3sequential_8/dense_76/Tensordot/GatherV2_1:output:00sequential_8/dense_76/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_8/dense_76/Tensordot/Prod_1?
+sequential_8/dense_76/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_8/dense_76/Tensordot/concat/axis?
&sequential_8/dense_76/Tensordot/concatConcatV2-sequential_8/dense_76/Tensordot/free:output:0-sequential_8/dense_76/Tensordot/axes:output:04sequential_8/dense_76/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_76/Tensordot/concat?
%sequential_8/dense_76/Tensordot/stackPack-sequential_8/dense_76/Tensordot/Prod:output:0/sequential_8/dense_76/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/dense_76/Tensordot/stack?
)sequential_8/dense_76/Tensordot/transpose	Transpose*layer_normalization_16/batchnorm/add_1:z:0/sequential_8/dense_76/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_8/dense_76/Tensordot/transpose?
'sequential_8/dense_76/Tensordot/ReshapeReshape-sequential_8/dense_76/Tensordot/transpose:y:0.sequential_8/dense_76/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_8/dense_76/Tensordot/Reshape?
&sequential_8/dense_76/Tensordot/MatMulMatMul0sequential_8/dense_76/Tensordot/Reshape:output:06sequential_8/dense_76/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_8/dense_76/Tensordot/MatMul?
'sequential_8/dense_76/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_76/Tensordot/Const_2?
-sequential_8/dense_76/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_76/Tensordot/concat_1/axis?
(sequential_8/dense_76/Tensordot/concat_1ConcatV21sequential_8/dense_76/Tensordot/GatherV2:output:00sequential_8/dense_76/Tensordot/Const_2:output:06sequential_8/dense_76/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_8/dense_76/Tensordot/concat_1?
sequential_8/dense_76/TensordotReshape0sequential_8/dense_76/Tensordot/MatMul:product:01sequential_8/dense_76/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_8/dense_76/Tensordot?
,sequential_8/dense_76/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_76_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_8/dense_76/BiasAdd/ReadVariableOp?
sequential_8/dense_76/BiasAddBiasAdd(sequential_8/dense_76/Tensordot:output:04sequential_8/dense_76/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_76/BiasAdd?
sequential_8/dense_76/ReluRelu&sequential_8/dense_76/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_76/Relu?
.sequential_8/dense_77/Tensordot/ReadVariableOpReadVariableOp7sequential_8_dense_77_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype020
.sequential_8/dense_77/Tensordot/ReadVariableOp?
$sequential_8/dense_77/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2&
$sequential_8/dense_77/Tensordot/axes?
$sequential_8/dense_77/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2&
$sequential_8/dense_77/Tensordot/free?
%sequential_8/dense_77/Tensordot/ShapeShape(sequential_8/dense_76/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_8/dense_77/Tensordot/Shape?
-sequential_8/dense_77/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_77/Tensordot/GatherV2/axis?
(sequential_8/dense_77/Tensordot/GatherV2GatherV2.sequential_8/dense_77/Tensordot/Shape:output:0-sequential_8/dense_77/Tensordot/free:output:06sequential_8/dense_77/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(sequential_8/dense_77/Tensordot/GatherV2?
/sequential_8/dense_77/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 21
/sequential_8/dense_77/Tensordot/GatherV2_1/axis?
*sequential_8/dense_77/Tensordot/GatherV2_1GatherV2.sequential_8/dense_77/Tensordot/Shape:output:0-sequential_8/dense_77/Tensordot/axes:output:08sequential_8/dense_77/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2,
*sequential_8/dense_77/Tensordot/GatherV2_1?
%sequential_8/dense_77/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential_8/dense_77/Tensordot/Const?
$sequential_8/dense_77/Tensordot/ProdProd1sequential_8/dense_77/Tensordot/GatherV2:output:0.sequential_8/dense_77/Tensordot/Const:output:0*
T0*
_output_shapes
: 2&
$sequential_8/dense_77/Tensordot/Prod?
'sequential_8/dense_77/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_77/Tensordot/Const_1?
&sequential_8/dense_77/Tensordot/Prod_1Prod3sequential_8/dense_77/Tensordot/GatherV2_1:output:00sequential_8/dense_77/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2(
&sequential_8/dense_77/Tensordot/Prod_1?
+sequential_8/dense_77/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_8/dense_77/Tensordot/concat/axis?
&sequential_8/dense_77/Tensordot/concatConcatV2-sequential_8/dense_77/Tensordot/free:output:0-sequential_8/dense_77/Tensordot/axes:output:04sequential_8/dense_77/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_77/Tensordot/concat?
%sequential_8/dense_77/Tensordot/stackPack-sequential_8/dense_77/Tensordot/Prod:output:0/sequential_8/dense_77/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/dense_77/Tensordot/stack?
)sequential_8/dense_77/Tensordot/transpose	Transpose(sequential_8/dense_76/Relu:activations:0/sequential_8/dense_77/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????( 2+
)sequential_8/dense_77/Tensordot/transpose?
'sequential_8/dense_77/Tensordot/ReshapeReshape-sequential_8/dense_77/Tensordot/transpose:y:0.sequential_8/dense_77/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2)
'sequential_8/dense_77/Tensordot/Reshape?
&sequential_8/dense_77/Tensordot/MatMulMatMul0sequential_8/dense_77/Tensordot/Reshape:output:06sequential_8/dense_77/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&sequential_8/dense_77/Tensordot/MatMul?
'sequential_8/dense_77/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_8/dense_77/Tensordot/Const_2?
-sequential_8/dense_77/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_8/dense_77/Tensordot/concat_1/axis?
(sequential_8/dense_77/Tensordot/concat_1ConcatV21sequential_8/dense_77/Tensordot/GatherV2:output:00sequential_8/dense_77/Tensordot/Const_2:output:06sequential_8/dense_77/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(sequential_8/dense_77/Tensordot/concat_1?
sequential_8/dense_77/TensordotReshape0sequential_8/dense_77/Tensordot/MatMul:product:01sequential_8/dense_77/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????( 2!
sequential_8/dense_77/Tensordot?
,sequential_8/dense_77/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_8/dense_77/BiasAdd/ReadVariableOp?
sequential_8/dense_77/BiasAddBiasAdd(sequential_8/dense_77/Tensordot:output:04sequential_8/dense_77/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2
sequential_8/dense_77/BiasAdd?
dropout_17/IdentityIdentity&sequential_8/dense_77/BiasAdd:output:0*
T0*+
_output_shapes
:?????????( 2
dropout_17/Identity?
add_1AddV2*layer_normalization_16/batchnorm/add_1:z:0dropout_17/Identity:output:0*
T0*+
_output_shapes
:?????????( 2
add_1?
5layer_normalization_17/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_17/moments/mean/reduction_indices?
#layer_normalization_17/moments/meanMean	add_1:z:0>layer_normalization_17/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2%
#layer_normalization_17/moments/mean?
+layer_normalization_17/moments/StopGradientStopGradient,layer_normalization_17/moments/mean:output:0*
T0*+
_output_shapes
:?????????(2-
+layer_normalization_17/moments/StopGradient?
0layer_normalization_17/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_17/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????( 22
0layer_normalization_17/moments/SquaredDifference?
9layer_normalization_17/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_17/moments/variance/reduction_indices?
'layer_normalization_17/moments/varianceMean4layer_normalization_17/moments/SquaredDifference:z:0Blayer_normalization_17/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????(*
	keep_dims(2)
'layer_normalization_17/moments/variance?
&layer_normalization_17/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_17/batchnorm/add/y?
$layer_normalization_17/batchnorm/addAddV20layer_normalization_17/moments/variance:output:0/layer_normalization_17/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????(2&
$layer_normalization_17/batchnorm/add?
&layer_normalization_17/batchnorm/RsqrtRsqrt(layer_normalization_17/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????(2(
&layer_normalization_17/batchnorm/Rsqrt?
3layer_normalization_17/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_17_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype025
3layer_normalization_17/batchnorm/mul/ReadVariableOp?
$layer_normalization_17/batchnorm/mulMul*layer_normalization_17/batchnorm/Rsqrt:y:0;layer_normalization_17/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_17/batchnorm/mul?
&layer_normalization_17/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/mul_1?
&layer_normalization_17/batchnorm/mul_2Mul,layer_normalization_17/moments/mean:output:0(layer_normalization_17/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/mul_2?
/layer_normalization_17/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_17_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype021
/layer_normalization_17/batchnorm/ReadVariableOp?
$layer_normalization_17/batchnorm/subSub7layer_normalization_17/batchnorm/ReadVariableOp:value:0*layer_normalization_17/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????( 2&
$layer_normalization_17/batchnorm/sub?
&layer_normalization_17/batchnorm/add_1AddV2*layer_normalization_17/batchnorm/mul_1:z:0(layer_normalization_17/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????( 2(
&layer_normalization_17/batchnorm/add_1?
IdentityIdentity*layer_normalization_17/batchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:?????????( 2

Identity?
NoOpNoOp0^layer_normalization_16/batchnorm/ReadVariableOp4^layer_normalization_16/batchnorm/mul/ReadVariableOp0^layer_normalization_17/batchnorm/ReadVariableOp4^layer_normalization_17/batchnorm/mul/ReadVariableOp<^multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp<^multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp>^multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp-^sequential_8/dense_76/BiasAdd/ReadVariableOp/^sequential_8/dense_76/Tensordot/ReadVariableOp-^sequential_8/dense_77/BiasAdd/ReadVariableOp/^sequential_8/dense_77/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????( : : : : : : : : : : : : : : : : 2b
/layer_normalization_16/batchnorm/ReadVariableOp/layer_normalization_16/batchnorm/ReadVariableOp2j
3layer_normalization_16/batchnorm/mul/ReadVariableOp3layer_normalization_16/batchnorm/mul/ReadVariableOp2b
/layer_normalization_17/batchnorm/ReadVariableOp/layer_normalization_17/batchnorm/ReadVariableOp2j
3layer_normalization_17/batchnorm/mul/ReadVariableOp3layer_normalization_17/batchnorm/mul/ReadVariableOp2z
;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_72/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_72/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_73/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_73/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_74/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_74/Tensordot/ReadVariableOp2z
;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp;multi_head_self_attention_8/dense_75/BiasAdd/ReadVariableOp2~
=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp=multi_head_self_attention_8/dense_75/Tensordot/ReadVariableOp2\
,sequential_8/dense_76/BiasAdd/ReadVariableOp,sequential_8/dense_76/BiasAdd/ReadVariableOp2`
.sequential_8/dense_76/Tensordot/ReadVariableOp.sequential_8/dense_76/Tensordot/ReadVariableOp2\
,sequential_8/dense_77/BiasAdd/ReadVariableOp,sequential_8/dense_77/BiasAdd/ReadVariableOp2`
.sequential_8/dense_77/Tensordot/ReadVariableOp.sequential_8/dense_77/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????( 
 
_user_specified_nameinputs
?
s
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_4834486

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
 
_user_specified_nameinputs"?L
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
serving_default_aux_input:0?????????
;
input_90
serving_default_input_9:0?????????(>

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
trainable_variables
	variables
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
trainable_variables
	variables
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
trainable_variables
 	variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"regularization_losses
#trainable_variables
$	variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'bias
(regularization_losses
)trainable_variables
*	variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
?
,regularization_losses
-trainable_variables
.	variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

0kernel
1bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

<kernel
=bias
>regularization_losses
?trainable_variables
@	variables
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Bkernel
Cbias
Dregularization_losses
Etrainable_variables
F	variables
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
regularization_losses
_non_trainable_variables
trainable_variables
`layer_metrics
	variables
ametrics

blayers
clayer_regularization_losses
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
etrainable_variables
f	variables
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
N
embeddings
hregularization_losses
itrainable_variables
j	variables
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
regularization_losses
lnon_trainable_variables
trainable_variables
mlayer_metrics
	variables
nmetrics

olayers
player_regularization_losses
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
vtrainable_variables
w	variables
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
|trainable_variables
}	variables
~	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
axis
	[gamma
\beta
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis
	]gamma
^beta
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?regularization_losses
?trainable_variables
?	variables
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
regularization_losses
?non_trainable_variables
trainable_variables
?layer_metrics
 	variables
?metrics
?layers
 ?layer_regularization_losses
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
"regularization_losses
?non_trainable_variables
#trainable_variables
?layer_metrics
$	variables
?metrics
?layers
 ?layer_regularization_losses
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
(regularization_losses
?non_trainable_variables
)trainable_variables
?layer_metrics
*	variables
?metrics
?layers
 ?layer_regularization_losses
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
,regularization_losses
?non_trainable_variables
-trainable_variables
?layer_metrics
.	variables
?metrics
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_78/kernel
:@2dense_78/bias
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
2regularization_losses
?non_trainable_variables
3trainable_variables
?layer_metrics
4	variables
?metrics
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_79/kernel
:@2dense_79/bias
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
8regularization_losses
?non_trainable_variables
9trainable_variables
?layer_metrics
:	variables
?metrics
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_80/kernel
:@2dense_80/bias
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
>regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
@	variables
?metrics
?layers
 ?layer_regularization_losses
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
Dregularization_losses
?non_trainable_variables
Etrainable_variables
?layer_metrics
F	variables
?metrics
?layers
 ?layer_regularization_losses
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
H:F 26token_and_position_embedding_8/embedding_16/embeddings
H:F( 26token_and_position_embedding_8/embedding_17/embeddings
Q:O  2?transformer_block_8/multi_head_self_attention_8/dense_72/kernel
K:I 2=transformer_block_8/multi_head_self_attention_8/dense_72/bias
Q:O  2?transformer_block_8/multi_head_self_attention_8/dense_73/kernel
K:I 2=transformer_block_8/multi_head_self_attention_8/dense_73/bias
Q:O  2?transformer_block_8/multi_head_self_attention_8/dense_74/kernel
K:I 2=transformer_block_8/multi_head_self_attention_8/dense_74/bias
Q:O  2?transformer_block_8/multi_head_self_attention_8/dense_75/kernel
K:I 2=transformer_block_8/multi_head_self_attention_8/dense_75/bias
!:  2dense_76/kernel
: 2dense_76/bias
!:  2dense_77/kernel
: 2dense_77/bias
>:< 20transformer_block_8/layer_normalization_16/gamma
=:; 2/transformer_block_8/layer_normalization_16/beta
>:< 20transformer_block_8/layer_normalization_17/gamma
=:; 2/transformer_block_8/layer_normalization_17/beta
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
H
?0
?1
?2
?3
?4"
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
 "
trackable_list_wrapper
'
M0"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
?
dregularization_losses
?non_trainable_variables
etrainable_variables
?layer_metrics
f	variables
?metrics
?layers
 ?layer_regularization_losses
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
hregularization_losses
?non_trainable_variables
itrainable_variables
?layer_metrics
j	variables
?metrics
?layers
 ?layer_regularization_losses
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
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Okernel
Pbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Qkernel
Rbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Skernel
Tbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ukernel
Vbias
?regularization_losses
?trainable_variables
?	variables
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
uregularization_losses
?non_trainable_variables
vtrainable_variables
?layer_metrics
w	variables
?metrics
?layers
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Wkernel
Xbias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ykernel
Zbias
?regularization_losses
?trainable_variables
?	variables
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
{regularization_losses
?non_trainable_variables
|trainable_variables
?layer_metrics
}	variables
?metrics
?layers
 ?layer_regularization_losses
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
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
<
q0
r1
s2
t3"
trackable_list_wrapper
 "
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
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
?regularization_losses
?non_trainable_variables
?trainable_variables
?layer_metrics
?	variables
?metrics
?layers
 ?layer_regularization_losses
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
.
y0
z1"
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
&:$@2Adam/dense_78/kernel/m
 :@2Adam/dense_78/bias/m
&:$@@2Adam/dense_79/kernel/m
 :@2Adam/dense_79/bias/m
&:$@@2Adam/dense_80/kernel/m
 :@2Adam/dense_80/bias/m
):'@2Adam/main_output/kernel/m
#:!2Adam/main_output/bias/m
M:K 2=Adam/token_and_position_embedding_8/embedding_16/embeddings/m
M:K( 2=Adam/token_and_position_embedding_8/embedding_17/embeddings/m
V:T  2FAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/m
P:N 2DAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/m
V:T  2FAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/m
P:N 2DAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/m
V:T  2FAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/m
P:N 2DAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/m
V:T  2FAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/m
P:N 2DAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/m
&:$  2Adam/dense_76/kernel/m
 : 2Adam/dense_76/bias/m
&:$  2Adam/dense_77/kernel/m
 : 2Adam/dense_77/bias/m
C:A 27Adam/transformer_block_8/layer_normalization_16/gamma/m
B:@ 26Adam/transformer_block_8/layer_normalization_16/beta/m
C:A 27Adam/transformer_block_8/layer_normalization_17/gamma/m
B:@ 26Adam/transformer_block_8/layer_normalization_17/beta/m
(:& 2Adam/aux_output/kernel/v
": 2Adam/aux_output/bias/v
&:$@2Adam/dense_78/kernel/v
 :@2Adam/dense_78/bias/v
&:$@@2Adam/dense_79/kernel/v
 :@2Adam/dense_79/bias/v
&:$@@2Adam/dense_80/kernel/v
 :@2Adam/dense_80/bias/v
):'@2Adam/main_output/kernel/v
#:!2Adam/main_output/bias/v
M:K 2=Adam/token_and_position_embedding_8/embedding_16/embeddings/v
M:K( 2=Adam/token_and_position_embedding_8/embedding_17/embeddings/v
V:T  2FAdam/transformer_block_8/multi_head_self_attention_8/dense_72/kernel/v
P:N 2DAdam/transformer_block_8/multi_head_self_attention_8/dense_72/bias/v
V:T  2FAdam/transformer_block_8/multi_head_self_attention_8/dense_73/kernel/v
P:N 2DAdam/transformer_block_8/multi_head_self_attention_8/dense_73/bias/v
V:T  2FAdam/transformer_block_8/multi_head_self_attention_8/dense_74/kernel/v
P:N 2DAdam/transformer_block_8/multi_head_self_attention_8/dense_74/bias/v
V:T  2FAdam/transformer_block_8/multi_head_self_attention_8/dense_75/kernel/v
P:N 2DAdam/transformer_block_8/multi_head_self_attention_8/dense_75/bias/v
&:$  2Adam/dense_76/kernel/v
 : 2Adam/dense_76/bias/v
&:$  2Adam/dense_77/kernel/v
 : 2Adam/dense_77/bias/v
C:A 27Adam/transformer_block_8/layer_normalization_16/gamma/v
B:@ 26Adam/transformer_block_8/layer_normalization_16/beta/v
C:A 27Adam/transformer_block_8/layer_normalization_17/gamma/v
B:@ 26Adam/transformer_block_8/layer_normalization_17/beta/v
?2?
)__inference_model_8_layer_call_fn_4832221
)__inference_model_8_layer_call_fn_4833167
)__inference_model_8_layer_call_fn_4833231
)__inference_model_8_layer_call_fn_4832889?
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
"__inference__wrapped_model_4831523input_9	aux_input"?
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
D__inference_model_8_layer_call_and_return_conditional_losses_4833536
D__inference_model_8_layer_call_and_return_conditional_losses_4833855
D__inference_model_8_layer_call_and_return_conditional_losses_4832960
D__inference_model_8_layer_call_and_return_conditional_losses_4833031?
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
@__inference_token_and_position_embedding_8_layer_call_fn_4833864?
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
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_4833888?
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
5__inference_transformer_block_8_layer_call_fn_4833925
5__inference_transformer_block_8_layer_call_fn_4833962?
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
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_4834206
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_4834464?
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
<__inference_global_average_pooling1d_8_layer_call_fn_4834469
<__inference_global_average_pooling1d_8_layer_call_fn_4834474?
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
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_4834480
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_4834486?
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
,__inference_aux_output_layer_call_fn_4834495?
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
G__inference_aux_output_layer_call_and_return_conditional_losses_4834506?
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
/__inference_concatenate_8_layer_call_fn_4834512?
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
J__inference_concatenate_8_layer_call_and_return_conditional_losses_4834519?
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
*__inference_dense_78_layer_call_fn_4834528?
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
E__inference_dense_78_layer_call_and_return_conditional_losses_4834539?
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
*__inference_dense_79_layer_call_fn_4834548?
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
E__inference_dense_79_layer_call_and_return_conditional_losses_4834559?
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
*__inference_dense_80_layer_call_fn_4834568?
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
E__inference_dense_80_layer_call_and_return_conditional_losses_4834579?
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
-__inference_main_output_layer_call_fn_4834588?
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
H__inference_main_output_layer_call_and_return_conditional_losses_4834599?
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
%__inference_signature_wrapper_4833103	aux_inputinput_9"?
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
.__inference_sequential_8_layer_call_fn_4831615
.__inference_sequential_8_layer_call_fn_4834612
.__inference_sequential_8_layer_call_fn_4834625
.__inference_sequential_8_layer_call_fn_4831688?
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_4834682
I__inference_sequential_8_layer_call_and_return_conditional_losses_4834739
I__inference_sequential_8_layer_call_and_return_conditional_losses_4831702
I__inference_sequential_8_layer_call_and_return_conditional_losses_4831716?
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
*__inference_dense_76_layer_call_fn_4834748?
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
E__inference_dense_76_layer_call_and_return_conditional_losses_4834779?
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
*__inference_dense_77_layer_call_fn_4834788?
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
E__inference_dense_77_layer_call_and_return_conditional_losses_4834818?
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
"__inference__wrapped_model_4831523?NMOPQRSTUV[\WXYZ]^&'0167<=BCZ?W
P?M
K?H
!?
input_9?????????(
#? 
	aux_input?????????
? "m?j
2

aux_output$?!

aux_output?????????
4
main_output%?"
main_output??????????
G__inference_aux_output_layer_call_and_return_conditional_losses_4834506\&'/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? 
,__inference_aux_output_layer_call_fn_4834495O&'/?,
%?"
 ?
inputs????????? 
? "???????????
J__inference_concatenate_8_layer_call_and_return_conditional_losses_4834519?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
/__inference_concatenate_8_layer_call_fn_4834512vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
E__inference_dense_76_layer_call_and_return_conditional_losses_4834779dWX3?0
)?&
$?!
inputs?????????( 
? ")?&
?
0?????????( 
? ?
*__inference_dense_76_layer_call_fn_4834748WWX3?0
)?&
$?!
inputs?????????( 
? "??????????( ?
E__inference_dense_77_layer_call_and_return_conditional_losses_4834818dYZ3?0
)?&
$?!
inputs?????????( 
? ")?&
?
0?????????( 
? ?
*__inference_dense_77_layer_call_fn_4834788WYZ3?0
)?&
$?!
inputs?????????( 
? "??????????( ?
E__inference_dense_78_layer_call_and_return_conditional_losses_4834539\01/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? }
*__inference_dense_78_layer_call_fn_4834528O01/?,
%?"
 ?
inputs?????????
? "??????????@?
E__inference_dense_79_layer_call_and_return_conditional_losses_4834559\67/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? }
*__inference_dense_79_layer_call_fn_4834548O67/?,
%?"
 ?
inputs?????????@
? "??????????@?
E__inference_dense_80_layer_call_and_return_conditional_losses_4834579\<=/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? }
*__inference_dense_80_layer_call_fn_4834568O<=/?,
%?"
 ?
inputs?????????@
? "??????????@?
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_4834480{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
W__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_4834486`7?4
-?*
$?!
inputs?????????( 

 
? "%?"
?
0????????? 
? ?
<__inference_global_average_pooling1d_8_layer_call_fn_4834469nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
<__inference_global_average_pooling1d_8_layer_call_fn_4834474S7?4
-?*
$?!
inputs?????????( 

 
? "?????????? ?
H__inference_main_output_layer_call_and_return_conditional_losses_4834599\BC/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ?
-__inference_main_output_layer_call_fn_4834588OBC/?,
%?"
 ?
inputs?????????@
? "???????????
D__inference_model_8_layer_call_and_return_conditional_losses_4832960?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
!?
input_9?????????(
#? 
	aux_input?????????
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
D__inference_model_8_layer_call_and_return_conditional_losses_4833031?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
!?
input_9?????????(
#? 
	aux_input?????????
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
D__inference_model_8_layer_call_and_return_conditional_losses_4833536?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
"?
inputs/0?????????(
"?
inputs/1?????????
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
D__inference_model_8_layer_call_and_return_conditional_losses_4833855?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
"?
inputs/0?????????(
"?
inputs/1?????????
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
)__inference_model_8_layer_call_fn_4832221?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
!?
input_9?????????(
#? 
	aux_input?????????
p 

 
? "=?:
?
0?????????
?
1??????????
)__inference_model_8_layer_call_fn_4832889?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
!?
input_9?????????(
#? 
	aux_input?????????
p

 
? "=?:
?
0?????????
?
1??????????
)__inference_model_8_layer_call_fn_4833167?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
"?
inputs/0?????????(
"?
inputs/1?????????
p 

 
? "=?:
?
0?????????
?
1??????????
)__inference_model_8_layer_call_fn_4833231?NMOPQRSTUV[\WXYZ]^&'0167<=BCb?_
X?U
K?H
"?
inputs/0?????????(
"?
inputs/1?????????
p

 
? "=?:
?
0?????????
?
1??????????
I__inference_sequential_8_layer_call_and_return_conditional_losses_4831702vWXYZC?@
9?6
,?)
dense_76_input?????????( 
p 

 
? ")?&
?
0?????????( 
? ?
I__inference_sequential_8_layer_call_and_return_conditional_losses_4831716vWXYZC?@
9?6
,?)
dense_76_input?????????( 
p

 
? ")?&
?
0?????????( 
? ?
I__inference_sequential_8_layer_call_and_return_conditional_losses_4834682nWXYZ;?8
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
I__inference_sequential_8_layer_call_and_return_conditional_losses_4834739nWXYZ;?8
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
.__inference_sequential_8_layer_call_fn_4831615iWXYZC?@
9?6
,?)
dense_76_input?????????( 
p 

 
? "??????????( ?
.__inference_sequential_8_layer_call_fn_4831688iWXYZC?@
9?6
,?)
dense_76_input?????????( 
p

 
? "??????????( ?
.__inference_sequential_8_layer_call_fn_4834612aWXYZ;?8
1?.
$?!
inputs?????????( 
p 

 
? "??????????( ?
.__inference_sequential_8_layer_call_fn_4834625aWXYZ;?8
1?.
$?!
inputs?????????( 
p

 
? "??????????( ?
%__inference_signature_wrapper_4833103?NMOPQRSTUV[\WXYZ]^&'0167<=BCm?j
? 
c?`
0
	aux_input#? 
	aux_input?????????
,
input_9!?
input_9?????????("m?j
2

aux_output$?!

aux_output?????????
4
main_output%?"
main_output??????????
[__inference_token_and_position_embedding_8_layer_call_and_return_conditional_losses_4833888[NM*?'
 ?
?
x?????????(
? ")?&
?
0?????????( 
? ?
@__inference_token_and_position_embedding_8_layer_call_fn_4833864NNM*?'
 ?
?
x?????????(
? "??????????( ?
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_4834206vOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????( 
p 
? ")?&
?
0?????????( 
? ?
P__inference_transformer_block_8_layer_call_and_return_conditional_losses_4834464vOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????( 
p
? ")?&
?
0?????????( 
? ?
5__inference_transformer_block_8_layer_call_fn_4833925iOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????( 
p 
? "??????????( ?
5__inference_transformer_block_8_layer_call_fn_4833962iOPQRSTUV[\WXYZ]^7?4
-?*
$?!
inputs?????????( 
p
? "??????????( 