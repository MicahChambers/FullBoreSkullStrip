#ifndef REGISTERIO_H
#define REGISTERIO_H
//Because ITK has this CRAZY IO system, we have to "register" each input
//factory, here I will register all of them. Just create one of these
//as a global

class RegisterIO
{
	public:
	RegisterIO();
};
#endif //REGISTERIO_H
