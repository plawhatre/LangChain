from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableSequence, 
    RunnableParallel, 
    RunnableLambda
)

# Input
x = {'a': 'Alice', 'b': 1}

# Runnables
runnable1 = RunnableLambda(lambda x: {'b':'hello' + x['a']})
runnable2 =  RunnableLambda(lambda x: {'new_b': str(x['b']) + 'bye'})
runnable3 = RunnablePassthrough()

# Composition Primitives
chain1 = RunnableSequence(runnable1, runnable2, runnable3)
chain2 = RunnableParallel(first=runnable1, second=runnable2, third=runnable3)
print('Chain 1: ', chain1.invoke(x)) 
print('Chain 2: ', chain2.invoke(x))

# Composition Syntax
chain1_operator = runnable1 | runnable2 | runnable3
chain1_method = runnable1.pipe(runnable2).pipe(runnable3)
print('Runnable sequence using operator: ', chain1_operator.invoke(x))
print('Runnable sequence using method: ', chain1_method.invoke(x))

chain2 = {
    "first": runnable1,
    "second": runnable2,
    "third": runnable3
} | RunnablePassthrough()
print('Runnable parallel using dictionary: ', chain2.invoke(x))





