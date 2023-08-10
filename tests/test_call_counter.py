from optson.call_counter import (
    class_method_call_counter,
    func_call_counter,
    get_func_count,
    get_method_count,
    method_call_counter,
)


def test_call_counter():
    @class_method_call_counter
    class MyClass:
        def method1(self):
            pass

        def method2(self):
            pass

    class MyOtherClass:
        @method_call_counter
        def method1(self):
            pass

    @func_call_counter
    def my_function():
        pass

    assert get_func_count(my_function) == 0

    my_function()
    my_function()

    assert get_func_count(my_function) == 2

    my_instance1 = MyClass()
    my_instance2 = MyClass()

    my_instance1.method1()
    my_instance1.method1()
    my_instance1.method1()
    my_instance1.method1()
    my_instance1.method2()

    my_instance2.method1()
    my_instance2.method2()

    assert get_method_count(my_instance1.method1) == 4
    assert get_method_count(my_instance1.method2) == 1
    assert get_method_count(my_instance2.method2) == 1

    my_instance3 = MyOtherClass()
    assert get_method_count(my_instance3.method1) == 0
    my_instance3.method1()
    assert get_method_count(my_instance3.method1) == 1
