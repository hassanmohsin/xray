def min_max_val(model, axis=0):
    mask = np.sum(model, axis=axis) > 0
    m = np.cumsum(model, axis=axis)
    mx = np.argmax(m, axis=axis)
    inc = np.ones(3, dtype=int)
    inc[axis] = -1
    m = np.cumsum(model[::inc[0], ::inc[1], ::inc[2]], axis=axis)
    mn = model.shape[axis] - np.argmax(m, axis=axis) - 1
    return mn, mx, mask


plt.close('all')

for file in ['c3_bolt_long_metal.stl_300.npy', 'c3_cylinder_metal.stl_200.npy']:
    model = np.load(file)

    for a in range(3):
        mn, mx, mask = min_max_val(model, axis=a)
        fig, ax = plt.subplots(ncols=3)
        ax[0].imshow(mn)
        ax[1].imshow(mx)
        ax[2].imshow(mask)
        print(np.amin(mn), np.amax(mx))

m = np.argmax(model, axis=0)

m = np.cumsum(model, axis=0)
m = np.argmax(m, axis=0)

plt.figure()
plt.imshow(m)
