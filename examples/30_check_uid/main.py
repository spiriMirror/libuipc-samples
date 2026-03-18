from uipc import dev

cuid_info = dev.ConstitutionUIDInfo()
iuid_info = dev.ImplicitGeometryUIDInfo()

print('Constitution UID Info:')
print(cuid_info)
print('Implicit Geometry UID Info:')
print(iuid_info)

cuid_info.first_available_uid()
print(f'First available constitution UID: {cuid_info.first_available_uid()}')

cuid_info.check_uid_available(1001)
print(f'Is UID 1001 available? {cuid_info.check_uid_available(1001)}')

iuid_info.first_available_uid()
print(f'First available implicit geometry UID: {iuid_info.first_available_uid()}')

iuid_info.check_uid_available(2001)
print(f'Is UID 2001 available? {iuid_info.check_uid_available(2001)}')