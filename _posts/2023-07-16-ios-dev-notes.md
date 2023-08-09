---
layout: post
title: "iOS Development Notes"
author: "Yalun Hu"
categories: journal
tags: [Blog, iOS, Swift]

image: mountains.jpg
--- 

## 如何快速查看文档

在Xcode中按住Option然后点击对应的关键字或builtin types，便能快速跳转到对应的文档下。

## 类和结构体对比

![struct_vs_class](../assets/img/2023-07-16-ios-dev-notes/struct_vs_class.png)

Structures and classes in Swift have many things in common. Both can:

- Define properties to store values
- Define methods to provide functionality
- Define subscripts to provide access to their values using subscript syntax
- Define initializers to set up their initial state
- Be extended to expand their functionality beyond a default implementation
- Conform to protocols to provide standard functionality of a certain kind

For more information, see [Properties](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/properties), [Methods](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/methods), [Subscripts](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/subscripts), [Initialization](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/initialization), [Extensions](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/extensions), and [Protocols](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols).

Classes have additional capabilities that structures don’t have:

- Inheritance enables one class to inherit the characteristics of another.
- Type casting enables you to check and interpret the type of a class instance at runtime.
- De-initializers enable an instance of a class to free up any resources it has assigned.
- Reference counting allows more than one reference to a class instance.

For more information, see [Inheritance](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/inheritance), [Type Casting](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/typecasting), [Deinitialization](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/deinitialization), and [Automatic Reference Counting](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/automaticreferencecounting).

## 并发[Concurrency]

Declaration of the Asynchronous Function. 
```swift
func listPhotos(inGallery name: String) async -> [String] {
    let result = // ... some asynchronous networking code ...
    return result
}
```

write `await` in front of the call to mark the possible suspension point.

```swift
let photoNames = await listPhotos(inGallery: "Summer Vacation")
```

Because code with await needs to be able to suspend execution, only certain places in your program can call asynchronous functions or methods:
- Code in the body of an asynchronous function, method, or property.
- Code in the static main() method of a structure, class, or enumeration that’s marked with @main.
- Code in an unstructured child task.

## 协议[Protocols]

总结来看，协议为Swift提供了一种方便，灵活的对类型进行抽象描述的方式。

A *protocol* defines a blueprint of methods, properties, and other requirements that suit a particular task or piece of functionality. The protocol can then be *adopted* by a class, structure, or enumeration to provide an actual implementation of those requirements. Any type that satisfies the requirements of a protocol is said to *conform* to that protocol.

Multiple protocols can be listed, and are separated by commas:

```swift
struct SomeStructure: FirstProtocol, AnotherProtocol {
    // structure definition goes here
}
```

If a class has a superclass, list the superclass name before any protocols it adopts, followed by a comma:

```swift
class SomeClass: SomeSuperclass, FirstProtocol, AnotherProtocol {
    // class definition goes here
}
```

### Initializer Requirements

Protocols can require specific initializers to be implemented by conforming types.You can implement a protocol initializer requirement on a conforming class as either a designated initializer or a convenience initializer. In both cases, you must mark the initializer implementation with the **required** modifier:

```swift
protocol SomeProtocol {
    init(someParameter: Int)
}

class SomeClass: SomeProtocol {
    required init(someParameter: Int) {
        // initializer implementation goes here
    }
}
```

You don’t need to mark protocol initializer implementations with the `required` modifier on classes that are marked with the `final` modifier, because final classes can’t subclassed.

If a subclass overrides a designated initializer from a superclass, and also implements a matching initializer requirement from a protocol, mark the initializer implementation with both the `required` and `override`modifiers:

```swift
protocol SomeProtocol {
    init()
}


class SomeSuperClass {
    init() {
        // initializer implementation goes here
    }
}


class SomeSubClass: SomeSuperClass, SomeProtocol {
    // "required" from SomeProtocol conformance; "override" from SomeSuperClass
    required override init() {
        // initializer implementation goes here
    }
}
```

### Protocol As Types

尽管协议不提供任何关于功能的具体实现，但是Swift中的协议也是可以作为一种类型进行使用的。

## Generic[泛型]



## MVVM设计模式

![mvvm](../assets/img/2023-07-16-ios-dev-notes/mvvm_arch.png)

Model-View-ViewModel is a design paradigm. 

It must be adhered to for SwiftUI to work.

### Model

- UI Independent
- Data + Logic

### View

- The reflection of the Model
- Stateless
- Declared(只有“var body”决定了view是如何绘制的)
- Reactive (Always reacting efficiently to the change on the model)
- Automatically observes publications from ViewModel( or subscribe what they interested at from the ViewModel) .Pulls data and rebuild itself.

### ViewModel

- Binds View to Model(so the change on the model cause the view to react and get rebuilt)
- Interpreter (between Model and View). Help View code stay clean and neat.
- Gatekeeper.  
- Constantly noticing changes in the Model
- Publish a message globally once any change in the Model is noticed (avoid have any connection to any of the View that using it to access the Model)
- Processing User Intent(Change the Model based on the events occurs in the View)

### Rules in MVVM

- The View must always get data from the Model by asking it from the ViewModel.
- The ViewModel never stores the data for the Model in side of itself.





